def get_custom_css() -> str:
    """
    Returns custom CSS styles for the NexusRAG application to give it a
    modern, polished, visual engineering lab feel.
    """
    return """
    <style>
    /* Main container refinements */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1250px;
    }

    /* Header styling */
    .nexus-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .nexus-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .nexus-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Pipeline stage cards */
    .pipeline-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin: 16px 0;
    }
    .stage-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .dark .stage-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
    }
    .stage-title {
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stage-detail {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* Chunk cards */
    .chunk-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .dark .chunk-card {
        background-color: #0f172a;
        border: 1px solid #334155;
    }
    .chunk-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .dark .chunk-badge {
        background-color: #075985;
        color: #e0f2fe;
    }

    /* Provenance pill */
    .prov-pill-both {
        background-color: #dcfce7;
        color: #166534;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .prov-pill-sem {
        background-color: #e0e7ff;
        color: #3730a3;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .prov-pill-kw {
        background-color: #fef3c7;
        color: #92400e;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Delta badges */
    .rank-up {
        color: #16a34a;
        font-weight: 700;
    }
    .rank-down {
        color: #dc2626;
        font-weight: 700;
    }
    .rank-same {
        color: #64748b;
        font-weight: 600;
    }
    </style>
    """
