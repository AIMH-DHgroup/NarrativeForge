"""Offline contract tests. Providers are simulated; factual truth isn't proved by mocks."""
import copy
import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from eventweaver.kb.store import KnowledgeBase, SCHEMA
from eventweaver.kb.ingest import add_document
from eventweaver.kb.sections import detect_structure, save_structure, sections, relink
from eventweaver.kb.structural_plan import propose, export, apply, material
from eventweaver.kb.write import write_event, write_events, approve_event, import_events
from eventweaver.kb.verify import verify_event
from eventweaver.kb.report import coverage_report
from eventweaver.kb.export import export_for_automatic_workflow
from eventweaver.kb.structural_runtime import enriched_export


class Scripted:
    def __init__(self,*answers): self.answers=list(answers);self.prompts=[]
    def generate(self,prompt,**kwargs):
        self.prompts.append(prompt)
        if not self.answers: raise AssertionError('Unexpected model call')
        answer=self.answers.pop(0)
        if callable(answer): answer=answer(prompt)
        return SimpleNamespace(text=json.dumps(answer),error=None)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.kb=KnowledgeBase(self.root/'kb.sqlite')
    def tearDown(self):
        self.kb.close();self.temp.cleanup()
    def pdf(self,name,chapters,words=220,outline=True):
        path=self.root/name
        with pymupdf.open() as pdf:
            toc=[]
            for i,(title,text) in enumerate(chapters,1):
                page=pdf.new_page();page.insert_text((72,72),title,fontsize=16)
                page.insert_textbox((72,100,520,780),text,fontsize=11)
                toc.append([1,title,i])
            if outline: pdf.set_toc(toc)
            pdf.save(path)
        return add_document(self.kb,path,enrich=False,ocr_mode='never',passage_words=words)
    def claim(self,doc,text,index=None,page=1):
        ps=self.kb.q('SELECT * FROM passages WHERE doc_id=? AND version=? AND page_no=? ORDER BY seq',(doc.doc_id,doc.version,page))
        p=next((p for p in ps if text in p['text']),ps[0])
        count=self.kb.one('SELECT COUNT(*) FROM claims')[0]+1
        cid=f'{doc.doc_id}.v{doc.version}.c{count:04}'
        self.kb.run('INSERT INTO claims VALUES (?,?,?,?,?,?,?)',(cid,doc.doc_id,doc.version,text,'description','certain','active'))
        self.kb.run('INSERT INTO claims_fts VALUES (?,?)',(cid,text))
        self.kb.run('INSERT INTO claim_evidence VALUES (?,?)',(cid,p['passage_id']))
        return cid,p['passage_id']
    def plan(self,doc,**kw):
        return propose(self.kb,None,[doc.doc_id],policy='strict_section',**kw)
    def single(self,text='The inscription was found in Rome in 1920.'):
        d=self.pdf('A.pdf',[('Chapter 1 Inscription Alba',text)])
        cid,pid=self.claim(d,text)
        plan=self.plan(d);eid=plan['units'][0]['event_id']
        return d,cid,pid,plan,eid
    def writer(self,cid,pid,text,title='Inscription Alba'):
        data=dict(title=title,sentences=[dict(text=text,claim_ids=[cid],passage_ids=[pid])])
        return Scripted(data,data)
    def verifier(self,cid,pid,status='supported',coverage='covered'):
        check=dict(status=status,passage_ids=[pid],claim_ids=[cid],reason='Mock judgement')
        return Scripted(check,check,dict(coverage=[dict(claim_id=cid,status=coverage,reason='Mock coverage')]))

    def test_A_catalog_twenty_chapters(self):
        d=self.pdf('catalog.pdf',[(f'Chapter {i} Inscription {i}',f'Inscription {i} was found in Rome. Its stone is marble.') for i in range(1,21)])
        for i in range(1,21):
            self.claim(d,f'Inscription {i} was found in Rome.',page=i)
            self.claim(d,'Its stone is marble.',page=i)
        p=self.plan(d)
        self.assertEqual(len(p['units']),20)
        self.assertTrue(all(len(u['claim_ids'])==2 for u in p['units']))
        self.assertEqual(len(p['unassigned']),0)

    def test_B_long_chapter_paragraph_blocks_preserve_provenance(self):
        paragraphs=[' '.join(f'Inscription Alba has documented property {i}.' for i in range(start, start + 5))
                    for start in range(0, 30, 5)]
        path=self.root/'long.pdf'
        with pymupdf.open() as pdf:
            for index, paragraph in enumerate(paragraphs):
                page=pdf.new_page()
                page.insert_textbox((72,72,520,780),('Chapter 1 Alba\n\n' if index == 0 else '') + paragraph,
                                    fontsize=11)
            pdf.set_toc([[1,'Chapter 1 Alba',1]])
            pdf.save(path)
        d=add_document(self.kb,path,enrich=False,ocr_mode='never',passage_words=20)
        cids=[self.claim(d,t,page=page)[0] for page,paragraph in enumerate(paragraphs,1)
              for t in paragraph.split('. ') if t]
        p=self.plan(d)
        self.assertEqual(len(p['units']),6)
        self.assertEqual(set(c for u in p['units'] for c in u['claim_ids']),set(cids))
        self.assertEqual(len({u['section_ids'][0] for u in p['units']}),6)
        for unit in p['units']:
            section=self.kb.one('SELECT start_char,end_char FROM document_sections WHERE section_id=?', (unit['section_ids'][0],))
            self.assertLess(section['start_char'],section['end_char'])

    def test_C_semantic_and_preferred_split(self):
        d=self.pdf('two.pdf',[('Chapter 1 Two episodes','Alba was restored in 1900. Beta was moved in 2000.')])
        a,_=self.claim(d,'Alba was restored in 1900.');b,_=self.claim(d,'Beta was moved in 2000.')
        strict=self.plan(d);self.assertEqual(len(strict['units']),1)
        sid=strict['units'][0]['section_ids'][0]
        response={'units':[dict(title='Alba',section_ids=[sid],claim_ids=[a],rationale=a+' independent restoration'),dict(title='Beta',section_ids=[sid],claim_ids=[b],rationale=b+' independent move')]}
        for policy in ['section_preferred','semantic']:
            p=propose(self.kb,Scripted(response),[d.doc_id],policy=policy)
            self.assertEqual(len(p['units']),2)

    def test_D_selected_AB_excludes_C_every_stage(self):
        a,ca,pa,_,_=self.single();b=self.pdf('B.pdf',[('Chapter 1 Beta','Beta is marble.')]);cb,pb=self.claim(b,'Beta is marble.')
        c=self.pdf('C.pdf',[('Chapter 1 Secret','SECRET_C happened in 9999.')]);cc,pc=self.claim(c,'SECRET_C happened in 9999.')
        plan=propose(self.kb,None,[a.doc_id,b.doc_id],policy='strict_section')
        eid=plan['units'][0]['event_id'];w=self.writer(ca,pa,'The inscription was found in Rome in 1920.')
        write_event(self.kb,w,eid);v=self.verifier(ca,pa);verify_event(self.kb,v,eid)
        approve_event(self.kb,eid)
        result=export_for_automatic_workflow(self.kb,plan['narrative_id'],self.root/'out')
        payload=result['manifest'].read_text()
        for prompt in w.prompts+v.prompts:
            self.assertNotIn('SECRET_C',prompt);self.assertNotIn(pc,prompt)
        self.assertNotIn('SECRET_C',payload);self.assertNotIn(cc,payload);self.assertNotIn('C.pdf',payload)
        forged=copy.deepcopy(plan);forged['units'][0]['claim_ids'].append(cc)
        with self.assertRaises(ValueError): apply(self.kb,plan['narrative_id'],forged)

    def test_E_complementary_cross_pdf_manual_merge(self):
        a,ca,pa,_,_=self.single();b=self.pdf('B.pdf',[('Chapter 1 Alba','The inscription is marble.')]);cb,pb=self.claim(b,'The inscription is marble.')
        p=propose(self.kb,None,[a.doc_id,b.doc_id],policy='section_preferred')
        p['units'][0]['claim_ids']+=p['units'][1]['claim_ids'];p['units'][0]['section_ids']+=p['units'][1]['section_ids'];p['units'][0]['rationale']='Curator confirmed inscription Alba identity in both sources';p['units']=p['units'][:1]
        apply(self.kb,p['narrative_id'],p);p=export(self.kb,p['narrative_id']);self.assertEqual(len(p['units']),1)
        eid=p['units'][0]['event_id'];meta,cs,ps,ss,fp=material(self.kb,eid)
        self.assertEqual({c['claim_id'] for c in cs},{ca,cb});self.assertEqual(len(ss),2)

    def test_F_conflicting_dates_preserved(self):
        d=self.pdf('conflict.pdf',[('Chapter 1 Alba','One account dates the find to 1900. Another dates the find to 1901.')])
        a,pa=self.claim(d,'One account dates the find to 1900.');b,pb=self.claim(d,'Another dates the find to 1901.')
        p=self.plan(d);eid=p['units'][0]['event_id']
        data=dict(title='Alba',sentences=[dict(text='One account dates the find to 1900; another dates it to 1901.',claim_ids=[a,b],passage_ids=[pa,pb])])
        w=Scripted(data,data);write_event(self.kb,w,eid)
        row=self.kb.one('SELECT text FROM event_versions WHERE event_id=?',(eid,))
        self.assertIn('1900',row[0]);self.assertIn('1901',row[0]);self.assertIn('never resolve a conflict',w.prompts[0])

    def test_G_unstructured_fallback(self):
        d=self.pdf('plain.pdf',[('plain words','The inscription is marble.')],outline=False)
        # Equal font sizes avoid typography headings in this fixture.
        path=self.root/'plain.pdf'
        with pymupdf.open() as pdf:
            pg=pdf.new_page();pg.insert_text((72,72),'The inscription is marble.',fontsize=11);pdf.save(path)
        d=add_document(self.kb,path,enrich=False,ocr_mode='never')
        ss=detect_structure(self.kb,d.doc_id)
        self.assertEqual(ss[0]['method'],'fallback');self.assertTrue(ss[0]['generated_title'])

    def test_H_legacy_migration_repeatable(self):
        path=self.root/'old.sqlite';c=sqlite3.connect(path);c.executescript(SCHEMA)
        c.execute("INSERT INTO narratives VALUES ('Nold',NULL,'Preserved',1,'',0,0,'then')");c.commit();c.close()
        for _ in range(2):
            k=KnowledgeBase(path);self.assertEqual(k.one('SELECT title FROM narratives')[0],'Preserved');self.assertEqual(k.q('PRAGMA foreign_key_check'),[]);k.close()

    def test_I_legacy_writer_verify_export(self):
        d,cid,pid,p,eid=self.single()
        # Remove only structural metadata in an isolated DB, leaving legacy entities.
        with self.kb.tx() as conn:
            conn.execute('DELETE FROM unit_sections');conn.execute('DELETE FROM narrative_units');conn.execute('DELETE FROM structural_plan_versions');conn.execute('DELETE FROM structural_narratives')
        write_event(self.kb,SimpleNamespace(generate=lambda *a,**kw:SimpleNamespace(text=f'TITLE|Alba\nS|{cid}|{pid}|The inscription was found in Rome in 1920.',error=None)),eid)
        verifier=SimpleNamespace(generate=lambda *a,**kw:SimpleNamespace(text=f'CHECK|1|supported|{pid}|{cid}|none|documented',error=None))
        self.assertTrue(verify_event(self.kb,verifier,eid)['ok']);approve_event(self.kb,eid)
        r=export_for_automatic_workflow(self.kb,p['narrative_id'],self.root/'legacy')
        with r['csv'].open() as f: self.assertEqual(list(csv.reader(f))[0],['title','description'])

    def test_L_omitted_claim_blocks_approval(self):
        d,cid,pid,p,eid=self.single();write_event(self.kb,self.writer(cid,pid,'The inscription was found in Rome in 1920.'),eid)
        result=verify_event(self.kb,self.verifier(cid,pid,coverage='omitted'),eid)
        self.assertFalse(result['ok']);self.assertEqual(coverage_report(self.kb,p['narrative_id'])['summary']['omitted'],1)
        with self.assertRaises(ValueError): approve_event(self.kb,eid)
        approve_event(self.kb,eid,override=True)
        with self.assertRaises(ValueError):export_for_automatic_workflow(self.kb,p['narrative_id'],self.root/'bad')

    def test_M_soft_length_no_padding(self):
        d,cid,pid,p,eid=self.single();w=self.writer(cid,pid,'The inscription was found in Rome in 1920.');write_event(self.kb,w,eid)
        self.assertIn('Indicative words 300',w.prompts[0])
        warning=self.kb.one('SELECT warnings FROM structural_event_basis')[0];self.assertIn('outside soft length',warning)
        self.assertEqual(self.kb.one('SELECT text FROM event_versions')[0],'The inscription was found in Rome in 1920.')

    def test_N_update_and_reimport_preserve_old(self):
        d,cid,pid,p,eid=self.single();write_event(self.kb,self.writer(cid,pid,'The inscription was found in Rome in 1920.'),eid)
        verify_event(self.kb,self.verifier(cid,pid),eid);approve_event(self.kb,eid)
        with pymupdf.open() as pdf:
            pg=pdf.new_page();pg.insert_text((72,72),'Updated inscription text.');pdf.save(self.root/'A.pdf')
        updated=add_document(self.kb,self.root/'A.pdf',enrich=False,ocr_mode='never')
        self.assertEqual(updated.version,2);self.assertTrue(self.kb.one('SELECT stale_reason FROM events WHERE event_id=?',(eid,))[0])
        self.assertEqual(self.kb.one('SELECT approved_version FROM events WHERE event_id=?',(eid,))[0],1)
        self.assertTrue(sections(self.kb,d.doc_id,1))
        with self.assertRaises(ValueError):material(self.kb,eid)

    def test_O_determinism_late_association(self):
        d=self.pdf('late.pdf',[('Chapter 1 Alba','Alba is marble.')]);first=detect_structure(self.kb,d.doc_id)
        self.assertEqual(first,detect_structure(self.kb,d.doc_id,force=True))
        cid,pid=self.claim(d,'Alba is marble.');relink(self.kb,d.doc_id,d.version)
        self.assertTrue(self.kb.one('SELECT 1 FROM section_claims WHERE claim_id=?',(cid,)))
        self.assertEqual(first,detect_structure(self.kb,d.doc_id))
        self.assertEqual(add_document(self.kb,self.root/'late.pdf',enrich=False,ocr_mode='never').status,'unchanged')

    def test_same_page_boundaries_and_crossing_passage(self):
        d=self.pdf('same.pdf',[('Chapter 1 Alba','Alba is marble.\nChapter 2 Beta\nBeta is bronze.')],outline=False,words=500)
        # Explicit reviewed structure can locate two sections on the same page.
        page=self.kb.one('SELECT text FROM pages WHERE doc_id=?',(d.doc_id,))[0];boundary=page.index('Chapter 2')
        base=dict(kind='chapter',level=1,start_seq=1,end_seq=1,method='curator',confidence=1,review_status='reviewed',generated_title=False,issues=[])
        ss=save_structure(self.kb,d.doc_id,[dict(base,title='Alba',start_char=0,end_char=boundary),dict(base,title='Beta',start_char=boundary,end_char=len(page))])
        ca,pa=self.claim(d,'Alba is marble.');cb,pb=self.claim(d,'Beta is bronze.');self.assertEqual(pa,pb)
        p=self.plan(d);self.assertEqual([len(u['claim_ids']) for u in p['units']],[1,1])
        self.assertEqual(self.kb.one('SELECT COUNT(*) FROM passages WHERE doc_id=?',(d.doc_id,))[0],1)

    def test_plan_revisions_stale_json_and_foreign_evidence(self):
        d,cid,pid,p,eid=self.single();write_event(self.kb,self.writer(cid,pid,'The inscription was found in Rome in 1920.'),eid)
        edited=copy.deepcopy(p);edited['units'][0]['title']='New curated title';apply(self.kb,p['narrative_id'],edited)
        self.assertTrue(self.kb.one('SELECT stale_reason FROM events WHERE event_id=?',(eid,))[0])
        with self.assertRaises(ValueError):apply(self.kb,p['narrative_id'],p)
        self.assertEqual(self.kb.one('SELECT COUNT(*) FROM structural_plan_versions')[0],2)
        c=self.pdf('C.pdf',[('Chapter 1 Secret','Foreign fact.')]);cc,pc=self.claim(c,'Foreign fact.')
        self.kb.run('INSERT INTO claim_evidence VALUES (?,?)',(cid,pc))
        with self.assertRaises(ValueError):material(self.kb,eid)

    def test_invalid_model_schema_and_context_fail_without_text(self):
        d,cid,pid,p,eid=self.single()
        bad=dict(title='bad',sentences=[dict(text='Invented',claim_ids=[cid],passage_ids=['FOREIGN'])])
        with self.assertRaises(ValueError):write_event(self.kb,Scripted(bad),eid)
        self.assertEqual(self.kb.one('SELECT COUNT(*) FROM event_versions')[0],0)
        p['config']['context_chars']=100;apply(self.kb,p['narrative_id'],p)
        with self.assertRaises(ValueError):write_event(self.kb,Scripted(),eid)
        self.assertEqual(self.kb.one('SELECT COUNT(*) FROM event_versions')[0],0)

    def test_theme_exclusions_accounted_for(self):
        d=self.pdf('theme.pdf',[('Chapter 1 Alba','Alba is marble. A ship reached Pisa.')]);a,pa=self.claim(d,'Alba is marble.');b,pb=self.claim(d,'A ship reached Pisa.')
        p=propose(self.kb,Scripted(dict(relevant=[a],excluded=[dict(claim_id=b,reason='Not about inscription')])),[d.doc_id],policy='strict_section',theme='Inscription')
        self.assertEqual(p['excluded'][0]['claim_id'],b);self.assertEqual(p['units'][0]['claim_ids'],[a])

    def test_hierarchy_structure_revision_blocks_old_plan(self):
        d,cid,pid,p,eid=self.single();ss=sections(self.kb,d.doc_id);ss[0]['review_status']='reviewed';save_structure(self.kb,d.doc_id,ss)
        with self.assertRaises(ValueError):material(self.kb,eid)
        self.assertEqual(self.kb.one('SELECT COUNT(*) FROM structure_revisions')[0],2)
        # A real parent/child hierarchy, entirely on the same stored page.
        page=self.kb.one('SELECT text FROM pages WHERE doc_id=?',(d.doc_id,))[0]
        start=page.index('The inscription')
        parent=dict(ss[0],start_char=0,end_char=len(page),level=1,title='Chapter 1 Alba')
        child=dict(ss[0],start_char=start,end_char=len(page),level=2,title='Alba details',kind='subsection')
        tree=save_structure(self.kb,d.doc_id,[parent,child])
        self.assertEqual(tree[1]['parent_id'],tree[0]['section_id'])
        overlap=dict(child,title='overlapping sibling',start_char=start+1)
        with self.assertRaises(ValueError):save_structure(self.kb,d.doc_id,[parent,child,overlap])

    def test_unsupported_numbers_and_empty_verifier(self):
        d,cid,pid,p,eid=self.single();write_event(self.kb,self.writer(cid,pid,'The inscription was found in Rome in 9999.'),eid)
        result=verify_event(self.kb,self.verifier(cid,pid),eid);self.assertFalse(result['ok'])
        with self.assertRaises(ValueError):verify_event(self.kb,Scripted({}),eid)

    def test_B_generation_thirty_passages_no_cutoff(self):
        facts=[f'Alba has documented property number {i}.' for i in range(30)]
        path=self.root/'thirty.pdf'
        with pymupdf.open() as pdf:
            for i,t in enumerate(facts):
                pg=pdf.new_page();pg.insert_text((72,72),('Chapter 1 Alba\n' if i==0 else '')+t)
            pdf.set_toc([[1,'Chapter 1 Alba',1]]);pdf.save(path)
        d=add_document(self.kb,path,enrich=False,ocr_mode='never')
        refs=[self.claim(d,t,page=i+1) for i,t in enumerate(facts)]
        p=self.plan(d,batch_claims=5);eid=p['units'][0]['event_id']
        sentences=[dict(text=t,claim_ids=[c],passage_ids=[pid]) for t,(c,pid) in zip(facts,refs)]
        inventories=[dict(title='Alba',sentences=sentences[i:i+5]) for i in range(0,30,5)]
        w=Scripted(*inventories,dict(title='Alba',sentences=sentences));write_event(self.kb,w,eid)
        self.assertEqual(len(self.kb.q('SELECT * FROM event_versions')),1)
        checks=[dict(status='supported',passage_ids=[refs[0][1]],claim_ids=[],reason='Title supported')]
        checks += [dict(status='supported',passage_ids=[pid],claim_ids=[cid],reason='Supported') for cid,pid in refs]
        coverage=[dict(coverage=[dict(claim_id=c,status='covered',reason='Conveyed') for c,_ in refs[i:i+5]]) for i in range(0,30,5)]
        verifier=Scripted(*checks,*coverage)
        self.assertTrue(verify_event(self.kb,verifier,eid)['ok'])
        self.assertEqual(coverage_report(self.kb,p['narrative_id'])['summary']['covered'],30)
        self.assertTrue(any(refs[-1][1] in prompt for prompt in verifier.prompts))

    def test_plan_import_unchanged_and_relation_evidence(self):
        d=self.pdf('relations.pdf',[('Chapter 1 Alba','Alba is marble.'),('Chapter 2 Beta','Beta is bronze.')])
        a,pa=self.claim(d,'Alba is marble.',page=1);b,pb=self.claim(d,'Beta is bronze.',page=2)
        p=self.plan(d);p['relations']=[dict(src=p['units'][0]['event_id'],dst=p['units'][1]['event_id'],kind='thematic',claim_ids=[a,b])]
        apply(self.kb,p['narrative_id'],p);p=export(self.kb,p['narrative_id'])
        self.assertEqual(len(p['relations']),1)
        apply(self.kb,p['narrative_id'],p)
        self.assertFalse(self.kb.one('SELECT stale_reason FROM events WHERE event_id=?',(p['units'][0]['event_id'],))[0])
        p=export(self.kb,p['narrative_id']);p['relations'][0]['claim_ids']=['FOREIGN']
        with self.assertRaises(ValueError):apply(self.kb,p['narrative_id'],p)

    def test_curator_version_requires_new_independent_verification(self):
        d,cid,pid,p,eid=self.single();write_event(self.kb,self.writer(cid,pid,'The inscription was found in Rome in 1920.'),eid)
        verify_event(self.kb,self.verifier(cid,pid),eid);approve_event(self.kb,eid)
        import_events(self.kb,p['narrative_id'],{'events':[dict(event_id=eid,title='Alba inscription',text='The inscription was found in Rome in 1920.') ]})
        with self.assertRaises(ValueError):export_for_automatic_workflow(self.kb,p['narrative_id'],self.root/'bad')
        verify_event(self.kb,self.verifier(cid,pid),eid);approve_event(self.kb,eid)
        self.assertEqual(self.kb.one('SELECT COUNT(*) FROM event_versions WHERE event_id=?',(eid,))[0],2)
        result=export_for_automatic_workflow(self.kb,p['narrative_id'],self.root/'good')
        manifest=json.loads(result['manifest'].read_text())
        self.assertEqual(manifest['events'][0]['source_documents'],['A.pdf'])
        self.assertEqual(manifest['events'][0]['row'],1)

    def test_I_corpus_pipeline_on_same_pdf(self):
        import re
        from unittest.mock import patch
        from eventweaver.narrative.pipeline import generate_corpus_narrative
        d,cid,pid,p,eid=self.single()
        class CorpusMock:
            def generate(inner,prompt,**kwargs):
                if 'SOURCE EVIDENCE:' in prompt:
                    evidence=re.findall(r'\[EVIDENCE ([^ ]+)',prompt)[0]
                    inner.evidence=evidence
                    text=f'EVENT|{evidence}|The inscription was found in Rome in 1920.'
                else:
                    text=f'PARAGRAPH|{inner.evidence}|The inscription was found in Rome in 1920.'
                return SimpleNamespace(text=text,error=None)
        with patch('eventweaver.narrative.pipeline.build_provider',return_value=CorpusMock()):
            result=generate_corpus_narrative(self.root/'A.pdf',output_dir=self.root/'corpus',provider_name='ollama',model='mock',ocr_mode='never',skip_verification=True)
        self.assertEqual(len(result),1)
        self.assertIn('1920',result[0].paragraph)

    def test_CLI_roundtrip_structure_plan_write_verify_export(self):
        from contextlib import redirect_stdout
        import io
        from unittest.mock import patch
        from eventweaver.cli import main
        d,cid,pid,p,eid=self.single()
        base=['kb','--kb',str(self.root/'kb.sqlite')]
        commands=[['sections-detect',d.doc_id],['sections-export',d.doc_id,'-o',str(self.root/'sections.json')],['sections-import',d.doc_id,str(self.root/'sections.json')],['plan-export',p['narrative_id'],'-o',str(self.root/'plan.json')],['plan-import',p['narrative_id'],str(self.root/'plan.json')],['report',p['narrative_id']]]
        with redirect_stdout(io.StringIO()):
            for cmd in commands:self.assertEqual(main(base+cmd),0)
            with patch('eventweaver.kb.cli._provider',return_value=self.writer(cid,pid,'The inscription was found in Rome in 1920.')):
                self.assertEqual(main(base+['write',p['narrative_id']]),0)
            with patch('eventweaver.kb.cli._provider',return_value=self.verifier(cid,pid)):
                self.assertEqual(main(base+['verify',p['narrative_id']]),0)
            self.assertEqual(main(base+['approve',eid]),0)
            self.assertEqual(main(base+['export-aw',p['narrative_id'],'--out-dir',str(self.root/'cli_out')]),0)
            self.assertEqual(main(base+['export-json',p['narrative_id'],'-o',str(self.root/'narrative.json')]),0)
        self.assertEqual(json.loads((self.root/'narrative.json').read_text())['events'][0]['source_documents'],['A.pdf'])


if __name__=='__main__':unittest.main()
