def get_custom_css() -> str:
    """
    Returns custom CSS styles for the NexusRAG application to give it a
    clean, ultra-modern, executive product feel.
    """
    return """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main container refinements */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Executive Top Header */
    .nexus-hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        color: #f8fafc;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .nexus-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nexus-hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        max-width: 720px;
        line-height: 1.5;
    }

    .system-status-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Active Document Header Bar */
    .active-doc-bar {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    }

    .active-doc-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .active-doc-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .meta-tag {
        background: rgba(51, 65, 85, 0.4);
        padding: 3px 9px;
        border-radius: 6px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    /* Hero Upload Card (When no document is active) */
    .hero-upload-card {
        background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
        border: 2px dashed rgba(99, 102, 241, 0.35);
        border-radius: 18px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }

    .hero-upload-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.15);
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 12px;
        display: inline-block;
    }

    /* Suggested Question Buttons */
    .suggested-section {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }

    .suggested-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Citations and Provenance */
    .citation-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.86rem;
    }

    .citation-badge {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }

    /* Lifecycle Trace Box */
    .lifecycle-trace-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* Clean Streamlit adjustments */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    }

    button[kind="secondary"] {
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
        background: #1e293b !important;
        color: #f1f5f9 !important;
        transition: all 0.15s ease !important;
    }

    button[kind="secondary"]:hover {
        border-color: #64748b !important;
        background: #334155 !important;
    }
    </style>
    """
