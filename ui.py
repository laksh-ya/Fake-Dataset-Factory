"""Minimal, theme-safe Gradio interface for Fake Dataset Factory."""

import time

import gradio as gr


UI_CSS = """
:root {
    --ui-bg: #f6f7f9;
    --ui-surface: #ffffff;
    --ui-surface-soft: #f0f3f6;
    --ui-text: #18212c;
    --ui-muted: #58697a;
    --ui-line: #dce2e8;
    --ui-accent: #1769aa;
    --ui-accent-soft: #e7f1f9;
    --ui-success: #147d4c;
    --ui-success-soft: #e8f6ee;
    --ui-shadow: 0 10px 30px rgba(25, 39, 52, .06);
}
.dark {
    --ui-bg: #0d1218;
    --ui-surface: #151c24;
    --ui-surface-soft: #1c2530;
    --ui-text: #edf2f7;
    --ui-muted: #9eacba;
    --ui-line: #2b3744;
    --ui-accent: #6aaddb;
    --ui-accent-soft: #172b3a;
    --ui-success: #6bc99a;
    --ui-success-soft: #142b21;
    --ui-shadow: 0 12px 34px rgba(0, 0, 0, .22);
}
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 20px 42px !important;
    color: var(--ui-text) !important;
    background: var(--ui-bg) !important;
}
.app-head {
    display: flex;
    align-items: center;
    gap: 13px;
    min-height: 48px;
}

.app-title {
    margin: 0 !important;
    color: var(--ui-text) !important;
    font-size: clamp(24px, 3vw, 32px) !important;
    font-weight: 720 !important;
    letter-spacing: -.035em !important;
    line-height: 1.1 !important;
}
.app-subtitle {
    margin: 5px 0 0 !important;
    color: var(--ui-muted) !important;
    font-size: 13px !important;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-left: auto;
    padding: 7px 10px;
    border: 1px solid var(--ui-line);
    border-radius: 999px;
    color: var(--ui-muted);
    background: var(--ui-surface);
    font-size: 11px;
    font-weight: 650;
    white-space: nowrap;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ui-success);
    box-shadow: 0 0 0 3px var(--ui-success-soft);
}
.topbar {
    align-items: center !important;
    gap: 14px !important;
    margin-bottom: 18px;
}
#project-results-btn {
    min-width: 156px;
}
#github-btn {
    min-width: 104px;
}
#project-results-btn button,
#github-btn a,
#github-btn button,
#clear-btn button {
    border-color: var(--ui-line) !important;
}
.project-card,
.surface {
    border: 1px solid var(--ui-line) !important;
    border-radius: 16px !important;
    background: var(--ui-surface) !important;
    box-shadow: var(--ui-shadow) !important;
}
.project-card {
    margin-bottom: 16px;
    padding: 6px 18px !important;
}
.project-head,
.result-row,
.model-strip,
.result-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
}
.project-head {
    padding: 13px 2px 10px;
    border-bottom: 1px solid var(--ui-line);
}
.project-head strong {
    color: var(--ui-text);
    font-size: 13px;
}
.project-head span,
.result-row span {
    color: var(--ui-muted);
    font-size: 11px;
}
.result-row {
    padding: 9px 2px;
    border-bottom: 1px solid var(--ui-line);
}
.result-row:last-child {
    border-bottom: 0;
}
.result-row strong {
    color: var(--ui-text);
    font-size: 12px;
    font-weight: 650;
}
.fid-value {
    color: var(--ui-accent) !important;
    font-variant-numeric: tabular-nums;
    font-weight: 720;
}
.workspace-row {
    align-items: stretch !important;
    gap: 16px !important;
}
.surface {
    padding: 20px !important;
}
.section-title {
    margin: 0 0 3px !important;
    color: var(--ui-text) !important;
    font-size: 16px !important;
    font-weight: 690 !important;
    letter-spacing: -.015em !important;
}
.section-note {
    margin: 0 0 15px !important;
    color: var(--ui-muted) !important;
    font-size: 11px !important;
}
.model-strip {
    min-height: 42px;
    margin: 2px 0 10px;
    padding: 9px 11px;
    border: 1px solid var(--ui-line);
    border-radius: 10px;
    background: var(--ui-surface-soft);
}
.model-strip strong {
    color: var(--ui-text);
    font-size: 12px;
}
.model-strip span {
    color: var(--ui-muted);
    font-size: 10px;
    text-align: right;
}
#generate-btn button {
    min-height: 47px;
    font-weight: 700;
}
.workspace {
    min-height: 520px;
}
.empty-state {
    display: grid;
    place-items: center;
    min-height: 330px;
    padding: 28px;
    border: 1px dashed var(--ui-line);
    border-radius: 12px;
    color: var(--ui-muted);
    background: var(--ui-surface-soft);
    text-align: center;
}
.empty-symbol {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    margin: 0 auto 10px;
    border: 1px solid var(--ui-line);
    border-radius: 11px;
    color: var(--ui-accent);
    background: var(--ui-surface);
    font-size: 18px;
}
.empty-state strong {
    display: block;
    color: var(--ui-text);
    font-size: 13px;
}
.empty-state span {
    display: block;
    margin-top: 4px;
    font-size: 11px;
}
.result-summary {
    margin-bottom: 12px;
    padding: 11px 12px;
    border: 1px solid var(--ui-line);
    border-radius: 10px;
    background: var(--ui-surface-soft);
}
.result-summary strong {
    display: block;
    color: var(--ui-text);
    font-size: 12px;
}
.result-summary span {
    color: var(--ui-muted);
    font-size: 10px;
}
.summary-metrics {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 6px;
}
.summary-metrics span {
    padding: 5px 7px;
    border: 1px solid var(--ui-line);
    border-radius: 7px;
    color: var(--ui-text);
    background: var(--ui-surface);
    font-variant-numeric: tabular-nums;
}
.error-summary {
    padding: 12px;
    border: 1px solid var(--ui-line);
    border-radius: 10px;
    color: var(--ui-text);
    background: var(--ui-surface-soft);
    font-size: 12px;
}
.gallery-shell {
    margin-top: 0 !important;
}
.action-row {
    align-items: end !important;
    gap: 10px !important;
    margin-top: 10px;
}
.dark .project-card,
.dark .surface,
.dark .status-badge,
.dark .model-strip,
.dark .empty-state,
.dark .result-summary,
.dark .summary-metrics span,
.dark .empty-symbol {
    color: var(--ui-text);
}
:focus-visible {
    outline: 3px solid color-mix(in srgb, var(--ui-accent) 38%, transparent) !important;
    outline-offset: 2px !important;
}
@media (max-width: 760px) {
    .gradio-container {
        padding: 18px 12px 28px !important;
    }
    .topbar {
        align-items: stretch !important;
    }
    .app-head {
        align-items: flex-start;
        flex-direction: column;
        gap: 9px;
    }
    .status-badge {
        margin-left: 0;
    }
    #project-results-btn,
    #github-btn {
        width: 100%;
    }
    .surface {
        padding: 15px !important;
    }
    .workspace {
        min-height: 0;
    }
    .empty-state {
        min-height: 220px;
    }
    .result-summary,
    .action-row {
        align-items: stretch !important;
        flex-direction: column;
    }
    .summary-metrics {
        justify-content: flex-start;
    }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        transition: none !important;
        animation: none !important;
    }
}
"""


def _theme():
    """Return one Gradio theme with deliberate light and dark values."""
    return gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f6f7f9",
        body_background_fill_dark="#0d1218",
        body_text_color="#18212c",
        body_text_color_dark="#edf2f7",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#151c24",
        background_fill_secondary="#f0f3f6",
        background_fill_secondary_dark="#1c2530",
        block_background_fill="#ffffff",
        block_background_fill_dark="#151c24",
        block_border_color="#dce2e8",
        block_border_color_dark="#2b3744",
        block_label_text_color="#526274",
        block_label_text_color_dark="#b2beca",
        block_title_text_color="#18212c",
        block_title_text_color_dark="#edf2f7",
        input_background_fill="#ffffff",
        input_background_fill_dark="#111820",
        input_border_color="#cfd7df",
        input_border_color_dark="#364453",
        input_placeholder_color="#7b8998",
        input_placeholder_color_dark="#8493a2",
        border_color_primary="#dce2e8",
        border_color_primary_dark="#2b3744",
        button_primary_background_fill="#1769aa",
        button_primary_background_fill_hover="#12598f",
        button_primary_background_fill_dark="#6aaddb",
        button_primary_background_fill_hover_dark="#80bae0",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#08131d",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f0f3f6",
        button_secondary_background_fill_dark="#1c2530",
        button_secondary_background_fill_hover_dark="#263240",
        button_secondary_text_color="#263442",
        button_secondary_text_color_dark="#edf2f7",
        shadow_drop="0 10px 30px rgba(25, 39, 52, 0.06)",
        shadow_drop_lg="0 16px 38px rgba(25, 39, 52, 0.08)",
        block_radius="12px",
        input_radius="10px",
        button_large_radius="10px",
    )


def build_app(model_configs, get_available_models, generate_images, device):
    """Build the compact project demo without changing inference behavior."""
    available = get_available_models()
    ready_count = sum(bool(ready) for ready in available.values())
    default_model = "dcgan" if available.get("dcgan") else next(
        (model_id for model_id, ready in available.items() if ready),
        None,
    )

    device_type = getattr(device, "type", str(device))
    device_label = {
        "mps": "Apple GPU (MPS)",
        "cuda": "NVIDIA GPU (CUDA)",
        "cpu": "CPU",
    }.get(device_type, str(device))

    choices = [
        (config["name"], model_id)
        for model_id, config in model_configs.items()
        if available.get(model_id)
    ]

    def image_limit(model_id):
        if model_id == "stable_diffusion":
            return 2
        if model_id in {"ddpm", "flow_matching_v2"}:
            return 4
        return 16

    def model_strip(model_id):
        if not model_id:
            return '<section class="model-strip"><strong>No model selected</strong></section>'
        config = model_configs[model_id]
        limit = image_limit(model_id)
        return (
            '<section class="model-strip">'
            f'<strong>{config["name"]}</strong>'
            f'<span>FID {config["fid"]:.2f} · up to {limit} images</span>'
            '</section>'
        )

    def empty_state():
        return """
        <section class="empty-state">
            <div><div class="empty-symbol">▦</div>
            <strong>Generated images will appear here</strong>
            <span>Choose a model and generate a batch.</span></div>
        </section>
        """

    rows = "".join(
        '<div class="result-row">'
        f'<strong>{config["name"]}</strong>'
        f'<span class="fid-value">{config["fid"]:.2f}</span>'
        '</div>'
        for config in model_configs.values()
    )
    project_results_html = (
        '<section>'
        '<div class="project-head"><strong>Established project results</strong>'
        '<span>FID · lower is better</span></div>'
        f'{rows}</section>'
    )

    def on_model_change(model_id):
        limit = image_limit(model_id)
        value = 1 if model_id == "stable_diffusion" else 2 if limit == 4 else 4
        return gr.update(maximum=limit, value=value), model_strip(model_id)

    def toggle_project_results(is_open):
        next_state = not is_open
        label = "Hide project results" if next_state else "Project results"
        return next_state, gr.update(visible=next_state), gr.update(value=label)

    def on_generate(model_id, target_class, count, evaluate):
        if not model_id:
            gr.Error("Select a model first.")
            return (
                gr.update(value=[], visible=False),
                '<div class="error-summary">Select a model first.</div>',
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

        started = time.perf_counter()
        try:
            images, fid, tstr, zip_path, status = generate_images(
                model_id,
                target_class,
                int(count),
                "Fast",
                evaluate == "Yes",
            )
            if not images:
                raise RuntimeError(status or "Generation failed")
        except Exception as exc:
            gr.Error(f"Generation failed: {exc}")
            safe_error = str(exc).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return (
                gr.update(value=[], visible=False),
                f'<div class="error-summary"><strong>Generation failed.</strong> {safe_error}</div>',
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )

        elapsed = time.perf_counter() - started
        config = model_configs[model_id]
        metrics = [f"FID {fid:.2f}"] if fid is not None else []
        if tstr is not None:
            metrics.append(f"TSTR {tstr:.1f}%")
        metric_html = "".join(f"<span>{metric}</span>" for metric in metrics)
        summary = (
            '<section class="result-summary">'
            f'<div><strong>{len(images)} images generated</strong>'
            f'<span>{config["name"]} · {target_class} · {elapsed:.1f}s</span></div>'
            f'<div class="summary-metrics">{metric_html}</div>'
            '</section>'
        )
        gallery_items = [
            (image, f"Image {index + 1}")
            for index, image in enumerate(images)
        ]
        return (
            gr.update(value=gallery_items, visible=True),
            summary,
            gr.update(value=zip_path, visible=True),
            gr.update(visible=True),
        )

    def clear_results():
        return (
            gr.update(value=[], visible=False),
            empty_state(),
            gr.update(value=None, visible=False),
            gr.update(visible=False),
        )

    header_html = f"""
    <header class="app-head">
        <div><h1 class="app-title">Fake Dataset Factory</h1>
        <p class="app-subtitle">Generate and evaluate synthetic chest X-rays.</p></div>
        <div class="status-badge"><span class="status-dot"></span>
        {ready_count}/{len(model_configs)} ready · {device_label}</div>
    </header>
    """

    with gr.Blocks(title="Fake Dataset Factory", css=UI_CSS, theme=_theme()) as app:
        results_open = gr.State(False)
        with gr.Row(elem_classes="topbar"):
            with gr.Column(scale=1, min_width=300):
                gr.HTML(header_html)
            results_button = gr.Button(
                "Project results",
                variant="secondary",
                scale=0,
                min_width=156,
                elem_id="project-results-btn",
            )
            gr.Button(
                "GitHub",
                variant="secondary",
                link="https://github.com/Laksh-ya/fake-Dataset-Factory",
                scale=0,
                min_width=104,
                elem_id="github-btn",
            )

        with gr.Group(visible=False, elem_classes="project-card") as results_panel:
            gr.HTML(project_results_html)

        with gr.Row(equal_height=False, elem_classes="workspace-row"):
            with gr.Column(scale=4, min_width=300, elem_classes=["surface", "controls"]):
                gr.HTML('<h2 class="section-title">Generate</h2><p class="section-note">Select a setup and run the model.</p>')
                model_dropdown = gr.Dropdown(
                    choices=choices,
                    value=default_model,
                    label="Model",
                    info="Six trained generators are available.",
                    interactive=True,
                )
                model_summary = gr.HTML(model_strip(default_model))
                target_class = gr.Radio(
                    choices=["Normal", "Pneumonia"],
                    value="Normal",
                    label="Class / reference",
                )
                count = gr.Slider(
                    minimum=1,
                    maximum=image_limit(default_model),
                    value=4,
                    step=1,
                    label="Number of images",
                )
                evaluate = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Run evaluation",
                )
                generate_button = gr.Button(
                    "Generate",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

            with gr.Column(scale=7, min_width=420, elem_classes=["surface", "workspace"]):
                gr.HTML('<h2 class="section-title">Output</h2><p class="section-note">64 × 64 grayscale PNG</p>')
                result_summary = gr.HTML(empty_state())
                gallery = gr.Gallery(
                    label="Generated images",
                    columns=4,
                    rows=2,
                    height=360,
                    object_fit="contain",
                    show_label=False,
                    visible=False,
                    elem_classes="gallery-shell",
                )
                with gr.Row(elem_classes="action-row"):
                    download = gr.File(
                        label="Download ZIP",
                        visible=False,
                        scale=3,
                    )
                    clear_button = gr.Button(
                        "Clear",
                        variant="secondary",
                        visible=False,
                        scale=1,
                        elem_id="clear-btn",
                    )

        model_dropdown.change(
            fn=on_model_change,
            inputs=model_dropdown,
            outputs=[count, model_summary],
            queue=False,
            api_name=False,
        )
        results_button.click(
            fn=toggle_project_results,
            inputs=results_open,
            outputs=[results_open, results_panel, results_button],
            queue=False,
            api_name=False,
        )
        generate_button.click(
            fn=on_generate,
            inputs=[model_dropdown, target_class, count, evaluate],
            outputs=[gallery, result_summary, download, clear_button],
            api_name="generate",
        )
        clear_button.click(
            fn=clear_results,
            outputs=[gallery, result_summary, download, clear_button],
            queue=False,
            api_name=False,
        )

    app.queue(default_concurrency_limit=1, max_size=8)
    return app
