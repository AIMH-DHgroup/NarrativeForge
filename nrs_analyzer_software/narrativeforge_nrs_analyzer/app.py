from __future__ import annotations

import tempfile
from pathlib import Path

from .analysis import AnalysisOptions, DEFAULT_BALANCED_WEIGHTS, run_analysis


def main():
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install dashboard dependencies with: pip install -e .[dashboard]"
        ) from exc

    st.set_page_config(page_title="NarrativeForge NRS Analyzer", layout="wide")
    st.title("NarrativeForge NRS Analyzer")
    st.write(
        "Upload a NarrativeForge benchmark ZIP or point the CLI at an extracted folder. "
        "The app generates NRS-focused tables, charts, Pareto analysis, and deployment tradeoff rankings."
    )

    uploaded = st.file_uploader("Upload experiments ZIP", type=["zip"])

    with st.sidebar:
        st.header("Analysis options")
        input_order = st.text_input("Input strategy order", "auto,brief,rag,full")
        prompt_order = st.text_input("Prompt strategy order", "short,standard,detailed")
        st.subheader("Balanced-score weights")
        w_nrs = st.slider("NRS", 0.0, 1.0, DEFAULT_BALANCED_WEIGHTS["NRS"], 0.01)
        w_speed = st.slider("Speed", 0.0, 1.0, DEFAULT_BALANCED_WEIGHTS["speed"], 0.01)
        w_size = st.slider("Size efficiency", 0.0, 1.0, DEFAULT_BALANCED_WEIGHTS["size_efficiency"], 0.01)
        w_stability = st.slider("Stability", 0.0, 1.0, DEFAULT_BALANCED_WEIGHTS["stability"], 0.01)
        w_reliability = st.slider("Reliability", 0.0, 1.0, DEFAULT_BALANCED_WEIGHTS["reliability"], 0.01)
        reliable_failure_max = st.slider("Reliable failure-rate cutoff", 0.0, 0.5, 0.05, 0.01)
        st.subheader("Coverage diagnostics")
        coverage_enabled = st.checkbox("Force coverage diagnostics", value=False)
        recompute_coverage = st.checkbox("Recompute coverage cache", value=False)
        coverage_threshold = st.slider("Coverage display threshold", 0.50, 0.95, 0.75, 0.05)
        skip_entity_coverage = st.checkbox("Disable entity coverage", value=False)
        skip_keyphrase_coverage = st.checkbox("Disable keyphrase coverage", value=False)
        case_detail_plots = st.checkbox("Generate per-case detail plots", value=False)
        run_button = st.button("Run analysis", type="primary")

    if uploaded is None:
        st.info("Upload a ZIP file to begin. For batch analysis, run `nrs-analyze experiments.zip -o output_dir` from a terminal.")
        return

    if not run_button:
        st.info("Set options in the sidebar and click Run analysis.")
        return

    with tempfile.TemporaryDirectory(prefix="nrs_dashboard_") as tmp:
        tmp = Path(tmp)
        source_path = tmp / uploaded.name
        source_path.write_bytes(uploaded.getvalue())
        output_dir = tmp / "analysis_output"
        options = AnalysisOptions(
            input_order=[v.strip() for v in input_order.split(",") if v.strip()],
            prompt_order=[v.strip() for v in prompt_order.split(",") if v.strip()],
            balanced_weights={
                "NRS": w_nrs,
                "speed": w_speed,
                "size_efficiency": w_size,
                "stability": w_stability,
                "reliability": w_reliability,
            },
            reliable_failure_rate_max=reliable_failure_max,
            coverage=coverage_enabled,
            coverage_thresholds=[0.70, float(coverage_threshold), 0.80],
            recompute_coverage=recompute_coverage,
            skip_entity_coverage=skip_entity_coverage,
            skip_keyphrase_coverage=skip_keyphrase_coverage,
            case_detail_plots=case_detail_plots,
        )

        with st.spinner("Running analysis..."):
            result = run_analysis(source_path, output_dir, options)

        summary = result.summary
        best_input = summary.get("best_input_strategy", {})
        best_prompt = summary.get("best_prompt_strategy", {})
        best_model = summary.get("best_model", {})
        global_best = summary.get("global_best_configuration", {})

        st.success("Analysis complete")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Runs", f"{summary.get('run_count', 0):,}")
        col2.metric("Models", f"{summary.get('model_count', 0):,}")
        col3.metric("Best input", str(best_input.get("input_strategy", "NA")), f"NRS {best_input.get('mean_NRS', 0):.2f}" if best_input else None)
        col4.metric("Best model", str(best_model.get("model", "NA")), f"NRS {best_model.get('mean_NRS', 0):.2f}" if best_model else None)

        st.subheader("Best raw configuration")
        if global_best:
            st.write(
                f"**{global_best.get('model')} + {global_best.get('input_strategy')} + {global_best.get('prompt_strategy')}** "
                f"— mean NRS {global_best.get('mean_NRS'):.3f}, runtime {global_best.get('mean_runtime_seconds'):.3f}s, "
                f"parameters {global_best.get('parameters_b'):.3f}B, failure rate {global_best.get('failure_rate'):.1%}."
            )

        st.subheader("Core tables")
        tab_names = [
            "descriptive_statistics",
            "input_strategy_summary",
            "prompt_strategy_summary",
            "input_prompt_summary",
            "model_summary",
            "best_tradeoff_configurations",
            "quality_loss_by_parameter_threshold",
            "pareto_optimal_configurations",
        ]
        tabs = st.tabs([name.replace("_", " ").title() for name in tab_names])
        for tab, name in zip(tabs, tab_names):
            with tab:
                st.dataframe(result.tables.get(name), use_container_width=True)

        st.subheader("Coverage Diagnostics")
        st.write(
            "BERTScore F1 and sentence-transformer semantic similarity compare complete generated narratives against complete source documents. "
            "Coverage diagnostics help identify cases where global semantic alignment may hide omitted facts or local details."
        )
        coverage_tab_names = [
            "coverage_by_input_strategy",
            "coverage_by_prompt_strategy",
            "coverage_by_model",
            "coverage_by_model_strategy_prompt",
            "coverage_by_case",
            "coverage_by_model_family",
            "high_alignment_low_coverage_cases",
        ]
        coverage_tabs = st.tabs([name.replace("_", " ").title() for name in coverage_tab_names])
        for tab, name in zip(coverage_tabs, coverage_tab_names):
            with tab:
                st.dataframe(result.tables.get(name), use_container_width=True)

        st.subheader("Figures")
        figure_items = summary.get("figure_interpretations", [])
        for item in figure_items:
            fig_path = result.figure_paths.get(item["figure"])
            if fig_path and fig_path.exists():
                st.markdown(f"### {item['title']}")
                st.image(str(fig_path), use_container_width=True)
                st.caption(item.get("interpretation", ""))
        st.subheader("Coverage figures")
        for key, fig_path in result.figure_paths.items():
            if "coverage" in key and fig_path.exists():
                st.markdown(f"### {key.replace('_', ' ').title()}")
                st.image(str(fig_path), use_container_width=True)

        st.subheader("Case Studies")
        st.write("Case-level diagnostics show whether aggregate benchmark conclusions are consistent across scenarios or driven by a few difficult cases.")
        case_tab_names = [
            "case_study_summary",
            "case_difficulty_ranking",
            "best_configuration_by_case",
            "case_input_strategy_delta",
            "case_prompt_strategy_delta",
            "case_model_size_loss",
        ]
        case_tabs = st.tabs([name.replace("_", " ").title() for name in case_tab_names])
        for tab, name in zip(case_tabs, case_tab_names):
            with tab:
                st.dataframe(result.tables.get(name), use_container_width=True)
        st.markdown("### Case-study figures")
        for key, fig_path in result.figure_paths.items():
            if key.startswith("case_") and fig_path.exists() and "per_case" not in str(fig_path):
                st.markdown(f"#### {key.replace('_', ' ').title()}")
                st.image(str(fig_path), use_container_width=True)
        case_summary = result.tables.get("case_study_summary")
        if case_summary is not None and not case_summary.empty:
            selected_case = st.selectbox("Select individual case study", case_summary["case_study"].astype(str).tolist())
            input_nrs = result.tables.get("case_by_input_strategy_nrs")
            prompt_nrs = result.tables.get("case_by_prompt_strategy_nrs")
            top5_case = result.tables.get("top5_configurations_by_case")
            if input_nrs is not None and not input_nrs.empty:
                st.markdown("#### NRS by input strategy")
                st.bar_chart(input_nrs[input_nrs["case_study"].astype(str) == selected_case].drop(columns=["case_study"]).T)
            if prompt_nrs is not None and not prompt_nrs.empty:
                st.markdown("#### NRS by prompt strategy")
                st.bar_chart(prompt_nrs[prompt_nrs["case_study"].astype(str) == selected_case].drop(columns=["case_study"]).T)
            if top5_case is not None and not top5_case.empty:
                st.markdown("#### Top configurations for selected case")
                st.dataframe(top5_case[top5_case["case_study"].astype(str) == selected_case], use_container_width=True)

        with open(result.bundle_path, "rb") as f:
            st.download_button(
                "Download complete analysis ZIP",
                f,
                file_name="narrativeforge_nrs_analysis_output.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
