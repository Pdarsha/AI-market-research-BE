import re
from typing import Optional

ROIC_METRICS_KEYWORDS = [
    "gross margin", "gross profit margin", "gross%", "gross_margin", "grossmargin", "gross margins", "gross profit margins",
    "operating margin", "operating profit margin", "ebit margin", "ebit_margin", "oper_margin", "operating_margin", "operatingmargin", "operating_profit_margin", "ebitmargin", "operating margins", "operating profit margins", "ebit margins",
    "ebitda margin", "ebitda_margin", "ebitdamargin", "ebitda margins",
    "ebitda", "ebitdavalue",
    "net margin", "net profit margin", "net_margin", "netmargin", "net profit margins", "net margins", "profit margin", "profit margins",
    "roic", "return_on_inv_capital", "returnoninvestedcapital", "return on invested capital",
    "roe", "return_com_eqy", "returnonequity", "return on equity",
    "roa", "return_on_asset", "returnonassets", "return on assets",
    "return on capital", "return_on_cap", "return on capitals",
    "free cash flow", "free_cash_flow", "cf_free_cash_flow", "fcf", "free cash flows",
    "fcf yield", "fcf_yield", "fcf yields",
    "pe ratio", "p/e ratio", "p/e", "pe", "pe_ratio", "pe ratios", "p/e ratios",
    "ev/ebitda", "ev / ebitda", "ev_to_ttm_ebitda",
    "price/sales", "price / sales", "p/s", "p/s ratio", "pr_to_sales_ratio", "price/sales ratio", "price/sales ratios",
    "enterprise value", "enterprise_value", "enterprise values",
    "market cap", "market_cap", "market capitalization", "market caps",
    "current ratio", "current_ratio", "cur_ratio", "current ratios",
    "quick ratio", "quick_ratio", "quick ratios",
    "debt/equity", "debt to equity", "debt-to-equity", "tot_debt_to_tot_eqy", "debt/equity ratio", "debt/equity ratios",
    "asset turnover", "asset_turnover", "asset turnovers",
    "inventory turnover", "inventory_turnover", "invent_turn", "inventory turnovers",
    "revenue growth", "revenue_growth", "revenue growths",
    "ebitda growth", "ebitda_growth", "ebitda growths",
    "eps growth", "eps_growth", "eps growths"
]

OVERVIEW_KEYWORDS = [
    "financial overview",
    "financial performance",
    "financials",
    "financial summary",
    "financial health",
    "financial analysis",
    "analyze financial",
]

DEFAULT_PHRASES = [
    "and highlight consensus and disagreements.",
    "and summarize what market participants are saying across the industry channel.",
    "covering the business model, financials, competitive position, and key risks.",
    "summarize guidance, growth drivers, margins, and key risks.",
    "covering funding history, business model, market, and growth outlook.",
    "about this company.",
    "Expert Interview Synthesizer",
    "Channel Check",
    "Earnings Analysis",
    "Company Primer",
    "Private Company Primer",
    "Top Investor Questions"
]

AGENT_FULL_TEMPLATES = {
    "Synthesize the key insights across the expert interviews": "Synthesize the key insights across the expert interviews and highlight consensus and disagreements.",
    "Run a channel check and summarize what market participants are saying": "Run a channel check and summarize what market participants are saying across the industry channel.",
    "Analyze the latest earnings call: summarize guidance": "Analyze the latest earnings call: summarize guidance, growth drivers, margins, and key risks.",
    "Conduct a company deep-dive primer covering the business model": "Conduct a company deep-dive primer covering the business model, financials, competitive position, and key risks.",
    "Conduct a private company primer covering funding history": "Conduct a private company primer covering funding history, business model, market, and growth outlook.",
    "Summarize the top investor questions and answers": "Summarize the top investor questions and answers about this company."
}

EXPLICIT_DOC_TERMS = (
    "my document",
    "uploaded document",
    "uploaded file",
    "uploaded report",
    "my report",
    "this report",
    "this pdf",
    "this file",
    "annual report",
    "page ",
    "compare my",
    "compare the report",
    "compare with my",
)

def has_roic_metric(content: str) -> bool:
    content_lower = content.lower()
    for keyword in ROIC_METRICS_KEYWORDS:
        if len(keyword) <= 4:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, content_lower):
                return True
        else:
            if keyword in content_lower:
                return True
    return False

def is_explicit_document_request(content: str) -> bool:
    content_lower = content.lower()
    return any(term in content_lower for term in EXPLICIT_DOC_TERMS)

def is_overview_request(content: str) -> bool:
    content_lower = content.lower()
    return any(term in content_lower for term in OVERVIEW_KEYWORDS)

def check_user_typed_context(content: str, matched_agent: Optional[dict]) -> bool:
    if matched_agent:
        prompt_sub = matched_agent["prompt_substring"]
        full_template = AGENT_FULL_TEMPLATES.get(prompt_sub, prompt_sub)
        cleaned_text = content.replace(full_template, "")
        cleaned_text = cleaned_text.replace(prompt_sub, "")
        for phrase in DEFAULT_PHRASES:
            cleaned_text = cleaned_text.replace(phrase, "")
        cleaned_text = cleaned_text.strip(" \t\n\r.,;:!?")
        return len(cleaned_text) > 5
    else:
        return len(content.strip()) > 5

def check_spurious_roic_query(roic_query: str, matched_agent: dict) -> bool:
    prompt_sub = matched_agent["prompt_substring"]
    full_template = AGENT_FULL_TEMPLATES.get(prompt_sub, prompt_sub)
    if prompt_sub in roic_query or full_template in roic_query:
        cleaned_query = roic_query.replace(full_template, "").replace(prompt_sub, "")
        for phrase in DEFAULT_PHRASES:
            cleaned_query = cleaned_query.replace(phrase, "")
        cleaned_query = cleaned_query.strip(" \t\n\r.,;:!?")
        if len(cleaned_query) <= 5:
            return True
    return False

def build_routing_log(content: str, document_ids: list, is_document_question: bool, selected_providers: list, skipped_providers: list) -> str:
    log_lines = [
        f"Question:\n{content}",
        f"Uploaded Documents:\n{len(document_ids)}",
        f"Document Relevant:\n{is_document_question}",
    ]
    if selected_providers:
        log_lines.append("Selected Providers:\n" + "\n".join(f"- {p}" for p in selected_providers))
    if skipped_providers:
        log_lines.append("Skipped:\n" + "\n".join(f"- {p}" for p in skipped_providers))
    return "\n\n".join(log_lines)
