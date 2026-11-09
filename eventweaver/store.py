"""Run-local storage of original document chunks, themes and narrative sections."""
import json
import sqlite3
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS chunks(id TEXT PRIMARY KEY, body TEXT);
        CREATE TABLE IF NOT EXISTS summaries(id TEXT PRIMARY KEY, level INTEGER, body TEXT,
                               raw_summary TEXT, byte_budget INTEGER, compression_method TEXT);
        CREATE TABLE IF NOT EXISTS summary_inputs(summary_id TEXT, input_id TEXT,
                                   PRIMARY KEY(summary_id, input_id));
        CREATE TABLE IF NOT EXISTS assignments(chunk_id TEXT, theme_id TEXT,
                                 PRIMARY KEY(chunk_id, theme_id));
        CREATE TABLE IF NOT EXISTS unassigned(chunk_id TEXT PRIMARY KEY, reason TEXT);
        CREATE TABLE IF NOT EXISTS paragraphs(theme_id TEXT, body TEXT);
        CREATE TABLE IF NOT EXISTS revisions(theme_id TEXT, level INTEGER, position INTEGER, body TEXT,
                               PRIMARY KEY(theme_id, level, position));
        CREATE TABLE IF NOT EXISTS responses(id INTEGER PRIMARY KEY AUTOINCREMENT, phase TEXT,
            call_number INTEGER, attempt INTEGER, response_text TEXT, error TEXT, truncated INTEGER);
        ''')

    def record_response(self, event):
        self.db.execute("INSERT INTO responses(phase,call_number,attempt,response_text,error,truncated) VALUES (?,?,?,?,?,?)",
                        (event["phase"], event["call_number"], event["attempt"], event["response_text"],
                         event["error"], int(event["truncated"])))
        self.db.commit()

    def rows(self, query, params=()):
        for row in self.db.execute(query, params):
            yield dict(row)

    def chunks(self, theme_id=None):
        query = "SELECT c.body FROM chunks c"
        params = ()
        if theme_id is not None:
            query += " JOIN assignments a ON a.chunk_id=c.id WHERE a.theme_id=?"
            params = (theme_id,)
        for row in self.rows(query + " ORDER BY c.id", params):
            yield json.loads(row["body"])

    def narrative_paragraphs(self, theme_id, level=None):
        if level is None:
            level = self.db.execute("SELECT max(level) FROM revisions WHERE theme_id=?", (theme_id,)).fetchone()[0]
        if level is None or level == -1:
            query = "SELECT body FROM paragraphs WHERE theme_id=? ORDER BY rowid"
            params = (theme_id,)
        else:
            query = "SELECT body FROM revisions WHERE theme_id=? AND level=? ORDER BY position"
            params = (theme_id, level)
        for row in self.rows(query, params):
            yield json.loads(row["body"])

    def export(self, directory: Path):
        queries = {
            "responses.jsonl": "SELECT * FROM responses ORDER BY id",
            "chunks.jsonl": "SELECT * FROM chunks ORDER BY id",
            "summaries.jsonl": "SELECT * FROM summaries ORDER BY level, id",
            "summary_inputs.jsonl": "SELECT * FROM summary_inputs ORDER BY summary_id, input_id",
            "assignments.jsonl": "SELECT * FROM assignments ORDER BY chunk_id, theme_id",
            "unassigned.jsonl": "SELECT * FROM unassigned ORDER BY chunk_id",
            "revisions.jsonl": "SELECT * FROM revisions ORDER BY theme_id, level, position",
            "paragraphs.jsonl": "SELECT * FROM paragraphs ORDER BY rowid",
        }
        for filename, query in queries.items():
            with (directory / filename).open("w", encoding="utf-8") as handle:
                for row in self.rows(query):
                    for key in ("body",):
                        if key in row:
                            row[key] = json.loads(row[key])
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
