import argparse

import gradio as gr

import webui
import webui_svc


def render_interface() -> gr.Blocks:
    with gr.Blocks(title="SoulX-Singer Preview", theme=gr.themes.Default()) as page:
        gr.HTML(
            '<div style="'
            'text-align: center; '
            'padding: 1.25rem 0 1.5rem; '
            'margin-bottom: 0.5rem;'
            '">'
            '<div style="'
            'display: inline-block; '
            'font-size: 1.9rem; '
            'font-weight: 700; '
            'letter-spacing: 0.02em; '
            'color: #1a1a2e; '
            'line-height: 1.3;'
            '">SoulX-Singer Preview</div>'
            '<div style="'
            'width: 96px; '
            'height: 3px; '
            'margin: 1rem auto 0; '
            'background: linear-gradient(90deg, transparent, #6366f1, transparent); '
            'border-radius: 2px;'
            '"></div>'
            '</div>'
        )
        with gr.Tabs():
            with gr.Tab("SVC"):
                webui_svc.build_interface()
            with gr.Tab("SVS"):
                webui.build_interface()
    return page


if __name__ == "__main__":
    print("This is a fusion app. Please start the application using uvicorn:")
    print("uvicorn api.app:app --host 0.0.0.0 --port 7860")
    import sys
    sys.exit(1)
