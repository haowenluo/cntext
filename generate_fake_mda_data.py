"""
Generate Fake MD&A Dataset for Testing
Creates realistic synthetic data matching the specifications:
- 5,576 filings
- Years 2008-2025
- Average MD&A length ~66,979 characters
- Includes financial terminology
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Real S&P 500 companies for realism
COMPANIES = [
    ("320193", "Apple Inc."),
    ("789019", "Microsoft Corporation"),
    ("1018724", "Amazon.com, Inc."),
    ("1652044", "Alphabet Inc."),
    ("1326801", "Meta Platforms, Inc."),
    ("1318605", "Tesla, Inc."),
    ("1045810", "NVIDIA Corporation"),
    ("19617", "JPMorgan Chase & Co."),
    ("886982", "Johnson & Johnson"),
    ("1067983", "Berkshire Hathaway Inc."),
    ("200406", "ExxonMobil Corporation"),
    ("1467373", "Visa Inc."),
    ("78003", "Pfizer Inc."),
    ("4962", "Amgen Inc."),
    ("1800", "Abbott Laboratories"),
    ("1090872", "Salesforce, Inc."),
    ("1113169", "Costco Wholesale Corporation"),
    ("1043219", "Broadcom Inc."),
    ("1091667", "Adobe Inc."),
    ("320187", "Cisco Systems, Inc."),
]

# Financial text templates with varied sentiment and topics
MD_A_TEMPLATES = [
    """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

OVERVIEW

The following discussion and analysis provides information which management believes is relevant to an assessment and understanding of our consolidated results of operations and financial condition. This discussion should be read in conjunction with the consolidated financial statements and notes thereto.

RESULTS OF OPERATIONS

Revenue for fiscal {year} increased by {growth_pct}% to ${revenue} million, compared to ${prev_revenue} million in the prior year. This growth was driven by strong performance across our product portfolio and expanding market share in key segments.

Gross margin improved to {gross_margin}% from {prev_gross_margin}% in the prior year, primarily due to favorable product mix, operational efficiencies, and economies of scale. However, we continue to face uncertainty regarding supply chain constraints and increasing input costs, which may negatively impact future margins.

Operating expenses increased by {opex_growth}% to ${opex} million, reflecting investments in research and development, sales and marketing initiatives, and infrastructure to support our growth. We remain committed to disciplined expense management while pursuing strategic opportunities.

Net income was ${net_income} million, or ${eps} per diluted share, compared to ${prev_net_income} million, or ${prev_eps} per diluted share, in the prior year. The increase reflects strong revenue growth and improved operational efficiency, partially offset by higher tax expenses.

LIQUIDITY AND CAPITAL RESOURCES

Cash and cash equivalents totaled ${cash} million at fiscal year-end, compared to ${prev_cash} million in the prior year. Operating activities generated ${operating_cf} million in cash flow, demonstrating our strong ability to convert earnings into cash.

We believe our existing cash balances, together with cash generated from operations, will be sufficient to satisfy our working capital needs, capital expenditures, and other liquidity requirements for at least the next 12 months. However, economic uncertainty, regulatory changes, and competitive pressures could adversely affect our liquidity position.

RISK FACTORS

Our business faces numerous risks and uncertainties, including but not limited to: macroeconomic conditions, competitive dynamics, regulatory compliance, cybersecurity threats, litigation risks, and potential disruptions to our supply chain. We have implemented comprehensive risk management programs, but cannot guarantee that these measures will be adequate.

The ongoing geopolitical tensions and trade policy changes create additional uncertainty for our international operations. We continue to monitor these developments closely and assess potential impacts on our business.

OUTLOOK

Looking ahead, we remain cautiously optimistic about our growth prospects. We plan to invest significantly in innovation, expand our market presence, and pursue strategic acquisitions that align with our long-term objectives. While we face headwinds from economic uncertainty and competitive pressures, we believe our strong market position, innovative products, and operational excellence will enable us to deliver sustainable value to our shareholders.

However, the current economic environment presents significant challenges. Inflation, rising interest rates, and potential recession risks create uncertainty about consumer spending and business investment. We are closely monitoring these trends and remain prepared to adjust our strategies as necessary.

We are subject to various legal proceedings and claims that arise in the ordinary course of business. While we believe we have meritorious defenses, litigation is inherently uncertain and adverse outcomes could have material impacts on our financial condition.

""",

    """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

EXECUTIVE SUMMARY

Our company delivered exceptional performance in fiscal {year}, achieving record revenues and profitability despite challenging market conditions. Strong execution of our strategic initiatives, combined with favorable market dynamics, drove outstanding results across all business segments.

FINANCIAL HIGHLIGHTS

Total revenue reached ${revenue} million, representing growth of {growth_pct}% compared to the prior year. This outstanding performance reflects robust demand for our products and services, successful new product launches, and market share gains in key categories.

Operating income increased {oi_growth}% to ${operating_income} million, with operating margin expanding to {operating_margin}% from {prev_operating_margin}% in the prior year. This margin expansion demonstrates our ability to drive operational efficiencies and leverage our scale advantages.

Diluted earnings per share grew {eps_growth}% to ${eps}, exceeding our guidance range and analyst expectations. This strong profitability reflects both revenue growth and improved cost management.

SEGMENT PERFORMANCE

Our core business segment generated revenue of ${segment1_rev} million, up {seg1_growth}% year-over-year. Growth was driven by increased unit volumes, favorable pricing, and expanding customer adoption of our premium offerings. However, we face increasing competition in this segment and must continue innovating to maintain our market leadership.

The emerging markets segment contributed ${segment2_rev} million in revenue, growing {seg2_growth}% compared to the prior year. While growth remains strong, we face risks related to currency fluctuations, regulatory changes, and political instability in certain regions.

OPERATIONAL EXCELLENCE

We made significant progress on operational initiatives during the year. Manufacturing efficiency improved through automation investments and process optimization. Supply chain resilience was enhanced through supplier diversification and inventory management improvements. These efforts position us well for continued growth, though we remain exposed to potential disruptions.

Research and development expenses increased to ${rd_expense} million as we accelerated innovation programs. We launched {num_products} new products during the year and expanded our intellectual property portfolio with {num_patents} new patents filed. These investments are critical to our long-term competitiveness, though there is no guarantee they will generate expected returns.

BALANCE SHEET AND CASH FLOW

Our balance sheet remains strong with total assets of ${total_assets} million and shareholders' equity of ${equity} million. Our debt-to-equity ratio of {debt_equity_ratio} provides financial flexibility while maintaining a conservative capital structure.

Cash flow from operations was ${operating_cf} million, up {cf_growth}% from the prior year. This strong cash generation enabled us to return ${shareholder_returns} million to shareholders through dividends and share repurchases, while continuing to invest in growth initiatives.

RISKS AND UNCERTAINTIES

We operate in a dynamic and competitive environment characterized by rapid technological change, evolving customer preferences, and regulatory complexity. Key risks include: cybersecurity threats, data privacy regulations, intellectual property litigation, product liability claims, and potential supply chain disruptions.

Economic uncertainty poses risks to consumer and business spending. Rising interest rates, inflation pressures, and geopolitical tensions could adversely affect demand for our products. We continuously assess these risks and adjust our strategies accordingly.

Environmental, social, and governance (ESG) factors increasingly impact our business. We face growing stakeholder expectations regarding sustainability, climate change mitigation, and social responsibility. Failure to meet these expectations could harm our reputation and financial performance.

FORWARD-LOOKING STATEMENTS

We expect continued growth in fiscal {next_year}, though at a more moderate pace than recent years. We plan significant investments in digital transformation, capacity expansion, and strategic partnerships. While we are optimistic about our prospects, numerous factors could cause actual results to differ materially from our expectations.

""",

    """Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations

BUSINESS OVERVIEW

We are a leading provider of innovative products and services serving diverse markets globally. Our business model focuses on customer value creation, operational excellence, and sustainable growth. Fiscal {year} presented both opportunities and challenges as we navigated evolving market conditions.

FINANCIAL PERFORMANCE

Fiscal {year} revenue totaled ${revenue} million, compared to ${prev_revenue} million in the prior year, representing growth of {growth_pct}%. While growth moderated from prior periods, we maintained solid market positions and made strategic investments for long-term value creation.

Gross profit was ${gross_profit} million with gross margin of {gross_margin}%, compared to {prev_gross_margin}% in the prior year. Margin pressure resulted from unfavorable product mix, competitive pricing dynamics, and increased input costs. We implemented cost reduction initiatives but face continued uncertainty regarding future margin trends.

Operating income declined to ${operating_income} million from ${prev_operating_income} million in the prior year. This decrease reflects both margin compression and increased operating expenses associated with transformation initiatives. We remain committed to improving operational efficiency and expect benefits from ongoing programs.

Net income attributable to shareholders was ${net_income} million, or ${eps} per diluted share, compared to ${prev_net_income} million, or ${prev_eps} per diluted share, in the prior year. Results were negatively impacted by operational challenges, restructuring charges, and unfavorable foreign exchange movements.

STRATEGIC INITIATIVES

We continued executing our multi-year transformation strategy focused on innovation, digital enablement, and portfolio optimization. Key accomplishments include: launching next-generation products, expanding our digital capabilities, and exiting non-core businesses. However, transformation programs involve significant risks and uncertainties, and we cannot guarantee successful outcomes.

During the year, we invested ${capex} million in capital expenditures to modernize facilities, expand capacity, and enhance technology infrastructure. These investments are essential for competitiveness but create near-term financial pressure and execution risks.

LIQUIDITY ANALYSIS

We ended the year with cash and cash equivalents of ${cash} million and total debt of ${total_debt} million. Our net debt position increased compared to the prior year due to acquisition funding and share repurchases. We maintain adequate liquidity to meet obligations, though our financial flexibility has diminished.

Operating cash flow was ${operating_cf} million, down from ${prev_operating_cf} million in the prior year. The decline reflects lower profitability and increased working capital requirements. We implemented working capital improvement programs but face ongoing challenges.

We have credit facilities totaling ${credit_facility} million with ${credit_available} million available at year-end. While we have sufficient liquidity for normal operations, deteriorating business conditions or credit market disruptions could constrain our access to financing.

REGULATORY AND LEGAL MATTERS

We operate in a complex regulatory environment with requirements related to product safety, environmental protection, data privacy, anti-corruption, and trade compliance. Regulatory requirements continue evolving, creating compliance challenges and potential liabilities.

We are involved in various legal proceedings, investigations, and claims, including product liability litigation, intellectual property disputes, and employment matters. While we defend vigorously, adverse outcomes could result in material financial impacts. We have established reserves based on current assessments, but actual liabilities could exceed reserves.

MARKET RISKS

Our business faces numerous market risks including: interest rate fluctuations, foreign currency volatility, commodity price changes, and equity market movements. We use derivative instruments to manage certain exposures but cannot eliminate all risks. Market disruptions could adversely affect our financial position and results.

Credit risk exists in our trade receivables and financial instruments. While we perform credit evaluations and maintain allowances for credit losses, economic deterioration could result in increased defaults and write-offs.

OUTLOOK AND CONCLUSION

Fiscal {next_year} will be challenging as we face economic headwinds, competitive pressures, and execution risks associated with our transformation programs. We are taking actions to improve performance, including cost reductions, operational improvements, and portfolio optimization.

While we remain committed to our long-term strategy, near-term results may be volatile. We continue monitoring market conditions closely and remain prepared to adjust plans as circumstances evolve. Success depends on numerous factors, many beyond our control.

"""
]

def generate_financial_values(year_idx, company_idx):
    """Generate realistic financial values with some correlation to year and company"""
    base_revenue = random.randint(5000, 150000)
    growth = random.uniform(-5, 25)

    values = {
        'year': 2008 + year_idx,
        'next_year': 2009 + year_idx,
        'revenue': base_revenue,
        'prev_revenue': int(base_revenue / (1 + growth/100)),
        'growth_pct': round(growth, 1),
        'gross_margin': round(random.uniform(35, 75), 1),
        'prev_gross_margin': round(random.uniform(33, 73), 1),
        'opex': int(base_revenue * random.uniform(0.25, 0.45)),
        'opex_growth': round(random.uniform(-3, 20), 1),
        'net_income': int(base_revenue * random.uniform(0.05, 0.25)),
        'prev_net_income': int(base_revenue * random.uniform(0.04, 0.23)),
        'eps': round(random.uniform(0.50, 15.00), 2),
        'prev_eps': round(random.uniform(0.45, 14.00), 2),
        'cash': int(base_revenue * random.uniform(0.15, 0.50)),
        'prev_cash': int(base_revenue * random.uniform(0.14, 0.48)),
        'operating_cf': int(base_revenue * random.uniform(0.15, 0.30)),
        'prev_operating_cf': int(base_revenue * random.uniform(0.14, 0.29)),
        'operating_income': int(base_revenue * random.uniform(0.10, 0.30)),
        'prev_operating_income': int(base_revenue * random.uniform(0.09, 0.29)),
        'operating_margin': round(random.uniform(12, 35), 1),
        'prev_operating_margin': round(random.uniform(11, 34), 1),
        'oi_growth': round(random.uniform(-10, 30), 1),
        'eps_growth': round(random.uniform(-5, 35), 1),
        'gross_profit': int(base_revenue * random.uniform(0.35, 0.65)),
        'segment1_rev': int(base_revenue * random.uniform(0.40, 0.70)),
        'seg1_growth': round(random.uniform(0, 20), 1),
        'segment2_rev': int(base_revenue * random.uniform(0.15, 0.35)),
        'seg2_growth': round(random.uniform(-5, 25), 1),
        'rd_expense': int(base_revenue * random.uniform(0.05, 0.20)),
        'num_products': random.randint(3, 25),
        'num_patents': random.randint(10, 200),
        'total_assets': int(base_revenue * random.uniform(1.5, 4.0)),
        'equity': int(base_revenue * random.uniform(0.5, 2.0)),
        'debt_equity_ratio': round(random.uniform(0.2, 1.5), 2),
        'shareholder_returns': int(base_revenue * random.uniform(0.05, 0.20)),
        'cf_growth': round(random.uniform(-8, 25), 1),
        'capex': int(base_revenue * random.uniform(0.08, 0.18)),
        'total_debt': int(base_revenue * random.uniform(0.3, 1.5)),
        'credit_facility': int(base_revenue * random.uniform(0.20, 0.50)),
        'credit_available': int(base_revenue * random.uniform(0.10, 0.30)),
    }
    return values

def generate_mda_text(template_idx, values):
    """Generate MD&A text by filling in template with values"""
    template = MD_A_TEMPLATES[template_idx % len(MD_A_TEMPLATES)]
    text = template.format(**values)

    # Add some variation in length to match statistics (avg ~67k chars)
    # Current templates are ~6-8k, so repeat sections with variations
    multiplier = random.randint(8, 12)

    additional_sections = []
    risk_phrases = [
        "We face significant risks related to market volatility and economic uncertainty.",
        "Litigation and regulatory proceedings could have material adverse effects on our business.",
        "Cybersecurity threats and data breaches pose ongoing risks to our operations.",
        "Supply chain disruptions could negatively impact our ability to meet customer demand.",
        "Climate change and environmental regulations may increase our operating costs.",
        "Talent acquisition and retention challenges could constrain our growth.",
        "Technological obsolescence risks require continuous innovation investments.",
        "Foreign currency fluctuations impact our international revenues and costs.",
        "Interest rate changes affect our borrowing costs and investment returns.",
        "Product recalls or quality issues could harm our reputation and financial performance.",
    ]

    for i in range(multiplier):
        section_type = random.choice(['risk', 'market', 'operations'])

        if section_type == 'risk':
            additional_sections.append(f"\n\nRISK CONSIDERATION {i+1}\n\n" + random.choice(risk_phrases) + " " * random.randint(50, 200))
        elif section_type == 'market':
            additional_sections.append(f"\n\nMARKET ANALYSIS {i+1}\n\nMarket conditions remained favorable with positive indicators across key segments. However, uncertainty persists regarding future trends. " + "Competition intensified during the period. " * random.randint(20, 50))
        else:
            additional_sections.append(f"\n\nOPERATIONAL UPDATE {i+1}\n\nOperational performance improved through continuous process enhancements and technology investments. Efficiency gains contributed to margin expansion. " + "We remain focused on operational excellence. " * random.randint(20, 50))

    full_text = text + "".join(additional_sections)
    return full_text

def generate_fake_data(num_records=5576, output_meta='mda_metadata.csv', output_full='mda_full.parquet'):
    """Generate fake MD&A dataset matching specifications"""

    print(f"Generating {num_records} fake MD&A filings...")

    # Distribute records across years 2008-2025 (18 years)
    years = list(range(2008, 2026))
    records_per_year = num_records // len(years)

    metadata_records = []
    full_records = []

    record_id = 0
    for year_idx, year in enumerate(years):
        # Determine how many records for this year
        if year == years[-1]:  # Last year gets remaining records
            year_records = num_records - record_id
        else:
            year_records = records_per_year

        for _ in range(year_records):
            # Pick a random company
            company_idx = record_id % len(COMPANIES)
            cik, company = COMPANIES[company_idx]

            # Generate filing date (random date in the following year after fiscal year end)
            fiscal_end_month = random.randint(1, 12)
            fiscal_end_day = random.randint(1, 28)
            period_of_report = f"{year}-{fiscal_end_month:02d}-{fiscal_end_day:02d}"

            # Filing date is typically 2-3 months after fiscal year end
            filing_offset_days = random.randint(60, 90)
            period_date = datetime.strptime(period_of_report, "%Y-%m-%d")
            filing_date = period_date + timedelta(days=filing_offset_days)
            filing_date_str = filing_date.strftime("%Y-%m-%d")

            # Generate financial values and MD&A text
            values = generate_financial_values(year_idx, company_idx)
            template_idx = random.randint(0, len(MD_A_TEMPLATES) - 1)
            mda_text = generate_mda_text(template_idx, values)

            # Create filename
            accession = f"{cik}-{year % 100}-{record_id:06d}"
            filename = f"{cik}_10K_{year}_{accession}.json"
            json_path = f"/data/sec_filings/{year}/{filename}"

            # Metadata record
            metadata_records.append({
                'filename': filename,
                'cik': cik,
                'company': company,
                'filing_date': filing_date_str,
                'period_of_report': period_of_report,
                'year': str(year),
                'has_item_7': True,
                'item_7_length': len(mda_text),
                'json_path': json_path
            })

            # Full record
            full_records.append({
                'cik': cik,
                'company': company,
                'filing_date': filing_date_str,
                'period_of_report': period_of_report,
                'year': str(year),
                'mda_text': mda_text
            })

            record_id += 1

            if record_id % 500 == 0:
                print(f"  Generated {record_id}/{num_records} records...")

    # Create DataFrames
    df_meta = pd.DataFrame(metadata_records)
    df_full = pd.DataFrame(full_records)

    # Save metadata CSV
    df_meta.to_csv(output_meta, index=False)
    print(f"\nSaved metadata to {output_meta}")
    print(f"  Shape: {df_meta.shape}")
    print(f"  Size: {df_meta.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Save full parquet
    df_full.to_parquet(output_full, compression='gzip', index=False)
    print(f"\nSaved full data to {output_full}")
    print(f"  Shape: {df_full.shape}")

    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total filings: {len(df_full):,}")
    print(f"Unique companies: {df_full['company'].nunique()}")
    print(f"Year range: {df_full['year'].min()} - {df_full['year'].max()}")
    print(f"\nMD&A Text Length Statistics:")
    print(f"  Mean: {df_full['mda_text'].str.len().mean():,.0f} characters")
    print(f"  Median: {df_full['mda_text'].str.len().median():,.0f} characters")
    print(f"  Min: {df_full['mda_text'].str.len().min():,} characters")
    print(f"  Max: {df_full['mda_text'].str.len().max():,} characters")
    print(f"\nFilings per year:")
    print(df_full['year'].value_counts().sort_index().to_string())
    print(f"\nTop companies by filing count:")
    print(df_full['company'].value_counts().head(10).to_string())

    return df_meta, df_full

if __name__ == '__main__':
    df_meta, df_full = generate_fake_data()
    print("\n✓ Fake dataset generation complete!")
