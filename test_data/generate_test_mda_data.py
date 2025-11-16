"""
Generate Realistic Test MD&A Data for Digitalization Attitudes Analysis

This script creates synthetic but realistic MD&A sections based on actual
patterns from real 10-K filings. The data includes companies with different
attitudes toward digitalization, AI, and technology adoption.
"""

import pandas as pd
import random
from datetime import datetime

# Set seed for reproducibility
random.seed(42)

# ============================================================================
# COMPANY PROFILES
# ============================================================================

companies = [
    # Digital Embracers - Positive toward AI/digitalization
    {
        'cik': '0000789019',
        'name': 'TechVanguard Inc.',
        'industry': 'Software',
        'digital_attitude': 'very_positive',
        'trend': 'increasing'
    },
    {
        'cik': '0001018724',
        'name': 'CloudFirst Corporation',
        'industry': 'Cloud Services',
        'digital_attitude': 'very_positive',
        'trend': 'stable'
    },
    {
        'cik': '0001045810',
        'name': 'AI Innovators Ltd.',
        'industry': 'AI/ML',
        'digital_attitude': 'very_positive',
        'trend': 'increasing'
    },
    {
        'cik': '0001652044',
        'name': 'DataStream Analytics',
        'industry': 'Analytics',
        'digital_attitude': 'positive',
        'trend': 'increasing'
    },
    {
        'cik': '0000320193',
        'name': 'NextGen Platforms',
        'industry': 'Technology',
        'digital_attitude': 'positive',
        'trend': 'stable'
    },

    # Balanced/Neutral Companies
    {
        'cik': '0000051143',
        'name': 'Balanced Manufacturing Co.',
        'industry': 'Manufacturing',
        'digital_attitude': 'neutral',
        'trend': 'increasing'
    },
    {
        'cik': '0000019617',
        'name': 'MidTech Industries',
        'industry': 'Industrial',
        'digital_attitude': 'neutral',
        'trend': 'stable'
    },
    {
        'cik': '0000886158',
        'name': 'Traditional Retail Group',
        'industry': 'Retail',
        'digital_attitude': 'neutral',
        'trend': 'increasing'
    },
    {
        'cik': '0000066740',
        'name': 'Regional Bank Holdings',
        'industry': 'Finance',
        'digital_attitude': 'neutral',
        'trend': 'decreasing'
    },
    {
        'cik': '0000093410',
        'name': 'Healthcare Systems Inc.',
        'industry': 'Healthcare',
        'digital_attitude': 'neutral',
        'trend': 'stable'
    },

    # Digital Skeptics - Cautious/Negative
    {
        'cik': '0000040545',
        'name': 'Legacy Industries Corp.',
        'industry': 'Manufacturing',
        'digital_attitude': 'negative',
        'trend': 'stable'
    },
    {
        'cik': '0000732717',
        'name': 'Conservative Energy LLC',
        'industry': 'Energy',
        'digital_attitude': 'negative',
        'trend': 'increasing'
    },
    {
        'cik': '0000882095',
        'name': 'Risk-Aware Financial',
        'industry': 'Finance',
        'digital_attitude': 'very_negative',
        'trend': 'stable'
    },
    {
        'cik': '0000021344',
        'name': 'Traditional Auto Group',
        'industry': 'Automotive',
        'digital_attitude': 'negative',
        'trend': 'decreasing'
    },
    {
        'cik': '0000354950',
        'name': 'Established Telecom Inc.',
        'industry': 'Telecommunications',
        'digital_attitude': 'neutral',
        'trend': 'decreasing'
    },
]

# ============================================================================
# MD&A TEXT TEMPLATES
# ============================================================================

# Very Positive Templates
very_positive_templates = [
    """
    Digital transformation continues to drive our strategic growth initiatives. During fiscal {year}, we accelerated our investment in artificial intelligence and machine learning capabilities, deploying advanced predictive analytics across our operations. Our cloud-first infrastructure has enabled unprecedented scalability and operational efficiency.

    We view digitalization as a tremendous opportunity to enhance customer experience and gain competitive advantage. Our AI-powered solutions have streamlined workflows, reduced operational costs by 15%, and improved decision-making across the organization. We are pioneering innovative applications of blockchain technology and exploring quantum computing partnerships.

    The adoption of intelligent automation has transformed our business model. We successfully implemented robotic process automation across 40% of our back-office functions, leveraging machine learning algorithms to optimize resource allocation. Our investment in data science talent and infrastructure positions us as an industry leader in digital innovation.

    Looking ahead, we plan to increase our technology budget by 25% to accelerate AI adoption, expand our digital platforms, and develop new intelligent products. We believe that embracing emerging technologies like IoT, edge computing, and 5G networks will unlock significant value for shareholders.
    """,

    """
    Innovation through digital technology remains our top strategic priority. This year, we deployed cutting-edge AI systems that enhanced our operational capabilities and customer engagement. Our machine learning models now process over 10 billion data points daily, enabling real-time insights and automated decision-making.

    We are enthusiastic about the potential of artificial intelligence to revolutionize our industry. Our deep learning algorithms have achieved breakthrough performance in predictive maintenance, reducing downtime by 30%. We view digitalization not as a risk, but as an essential enabler of growth and competitiveness.

    Our technology transformation initiatives delivered exceptional results. Cloud migration improved system reliability to 99.99% uptime while reducing infrastructure costs. We launched an AI-powered analytics platform that provides actionable intelligence to our teams. Investment in digital innovation grew 40% year-over-year.

    We are committed to maintaining our position as a technology leader. Planned initiatives include expanding our AI research team, implementing advanced automation across all business units, and developing next-generation digital products. These investments will strengthen our competitive moat and drive long-term value creation.
    """
]

# Positive Templates
positive_templates = [
    """
    Technology adoption has been a key focus area for our organization. In {year}, we implemented several digital initiatives to improve operational efficiency and customer service. Our new cloud-based systems have enhanced data accessibility and collaboration across teams.

    We recognize the growing importance of artificial intelligence and have begun integrating ML-powered tools into our workflows. These technologies show promise in optimizing our supply chain and improving forecasting accuracy. While we remain mindful of implementation challenges, we believe the benefits outweigh the risks.

    Our digital transformation journey is progressing steadily. We invested in upgrading our IT infrastructure, deploying advanced analytics capabilities, and training employees on new digital tools. E-commerce channels grew 20%, demonstrating the value of our online platform investments.

    Moving forward, we plan to continue modernizing our technology stack and exploring opportunities in automation and data analytics. We view digitalization as an important lever for growth, though we will proceed thoughtfully to ensure proper risk management and regulatory compliance.
    """,

    """
    Digital innovation contributed meaningfully to our performance this year. We adopted new software platforms that streamlined operations and improved productivity across multiple departments. Our technology initiatives focused on enhancing both efficiency and customer experience.

    Artificial intelligence and machine learning represent exciting opportunities for our business. We piloted several AI applications in customer service and operations, with encouraging early results. We are investing in building internal data science capabilities to leverage these technologies more effectively.

    Our modernization efforts included migrating core systems to the cloud, implementing advanced cybersecurity measures, and deploying mobile applications for field operations. These digital tools have enabled better decision-making through improved data visibility and analytics.

    We plan to accelerate our digital agenda in the coming year, with focused investments in automation, analytics, and digital channels. While we approach new technologies carefully, we believe strategic adoption of innovation will strengthen our competitive position.
    """
]

# Neutral Templates
neutral_templates = [
    """
    Technology infrastructure remained stable during {year}. We maintained our existing IT systems and made selective upgrades to ensure operational continuity. Our approach to digitalization balances innovation with prudent risk management and cost control.

    We continue to evaluate emerging technologies such as artificial intelligence and cloud computing. While these innovations show potential benefits, we also recognize implementation challenges, cybersecurity risks, and regulatory uncertainties. Our strategy emphasizes proven technologies with clear returns on investment.

    During the year, we implemented incremental improvements to our digital capabilities, including updates to our website, enhancements to internal systems, and limited automation of routine processes. These initiatives delivered modest efficiency gains while managing costs effectively.

    Looking ahead, we will continue to monitor technological developments and invest selectively in areas that align with our strategic priorities. We balance the opportunity for operational improvement through technology with the need to manage risks and maintain financial discipline.
    """,

    """
    Our technology operations focused on maintaining system reliability and meeting business requirements. We made necessary investments in infrastructure updates and security enhancements while containing IT expenses within budget targets.

    The company continues to assess opportunities in digital technologies. We recognize both the potential advantages and the challenges associated with AI, automation, and cloud services. Implementation requires significant investment, carries execution risks, and faces regulatory uncertainty in several jurisdictions.

    This year we completed standard system upgrades, refreshed hardware according to replacement cycles, and addressed cybersecurity vulnerabilities. Digital initiatives were evaluated based on clear business cases and expected payback periods. We adopted a measured approach to new technology deployment.

    Our technology roadmap emphasizes stability, security, and cost-effectiveness. While we monitor industry trends in digitalization and artificial intelligence, our focus remains on proven solutions that support core business operations with acceptable risk profiles.
    """
]

# Negative Templates
negative_templates = [
    """
    Technology investments were carefully controlled during {year} as we prioritized profitability and cash flow. While digital trends continue to evolve, we face significant challenges in AI implementation including high costs, talent scarcity, uncertain returns, and regulatory complexity.

    We remain concerned about cyber security risks and data privacy liabilities associated with increased digitalization. Recent high-profile breaches in our industry highlight the vulnerabilities of complex technology systems. We must balance innovation with robust risk management and compliance requirements.

    The rapid pace of technological change creates uncertainty for our traditional business model. Disruption from digital competitors, automation threats to our workforce, and substantial investment requirements constrain our ability to pursue aggressive technology transformation.

    Our approach emphasizes protecting our established market position and maintaining operational stability. While competitors pursue costly AI and cloud initiatives, we focus on incremental improvements to proven systems. We believe conservative technology strategy reduces risk and preserves capital for shareholders.
    """,

    """
    Digital transformation initiatives present significant challenges and risks for our organization. The costs of implementing advanced technologies like artificial intelligence remain prohibitive, with uncertain payback periods and potential for failed deployment.

    We are cautious about cloud migration due to data security concerns, compliance requirements, and dependency on third-party providers. Our industry faces heightened regulatory scrutiny around AI bias, algorithmic decision-making, and privacy protection. These regulatory uncertainties make aggressive technology adoption risky.

    Automation and AI could disrupt our workforce and require substantial retraining investments. Cybersecurity threats continue to escalate, with digitalization increasing our attack surface. We have experienced system vulnerabilities that reinforce the need for careful technology risk management.

    Our technology strategy prioritizes stability over innovation. We maintain legacy systems that, while older, provide reliable performance with understood risk profiles. We will continue to monitor digital trends but believe our conservative approach protects long-term shareholder value better than risky transformation initiatives.
    """
]

# Very Negative Templates
very_negative_templates = [
    """
    The company faces substantial risks from technology disruption and digitalization trends. During {year}, we encountered significant challenges with IT system failures, cybersecurity breaches, and costly technology project overruns that negatively impacted results.

    Artificial intelligence and automation pose serious threats to our traditional business model. We are deeply concerned about regulatory liability, algorithmic bias risks, data privacy violations, and potential job displacement. The costs and complexities of AI implementation far exceed any speculative benefits.

    Digital transformation has proven disruptive and value-destructive in our experience. Cloud migration attempts resulted in service outages and data integrity issues. Our industry faces existential threats from digital competitors that benefit from different regulatory treatment and lower cost structures.

    We believe aggressive technology spending destroys shareholder value through speculative investments with poor returns. Our focus is on cost reduction, risk mitigation, and defending our established market position. We will resist costly digital initiatives that lack clear profitability and expose the company to unacceptable liabilities and uncertainties.
    """,
]

# ============================================================================
# GENERATE MD&A CONTENT
# ============================================================================

def generate_mda_text(company, year):
    """Generate MD&A text based on company profile and year."""

    attitude = company['digital_attitude']
    trend = company['trend']

    # Base attitude determines template
    if attitude == 'very_positive':
        base_templates = very_positive_templates
        base_score = 0.8
    elif attitude == 'positive':
        base_templates = positive_templates
        base_score = 0.5
    elif attitude == 'neutral':
        base_templates = neutral_templates
        base_score = 0.0
    elif attitude == 'negative':
        base_templates = negative_templates
        base_score = -0.5
    else:  # very_negative
        base_templates = very_negative_templates
        base_score = -0.8

    # Adjust for trend over time
    year_offset = year - 2022  # 0, 1, or 2
    if trend == 'increasing':
        adjustment = year_offset * 0.15
    elif trend == 'decreasing':
        adjustment = year_offset * -0.15
    else:  # stable
        adjustment = 0

    # Select template
    template_idx = (int(company['cik']) + year) % len(base_templates)
    template = base_templates[template_idx]

    # Format with year
    text = template.format(year=year).strip()

    # Add some variation based on industry
    industry_specific = f"\n\nIndustry Context: Our {company['industry']} sector continues to evolve, presenting both opportunities and challenges for our business model."

    text += industry_specific

    return text

# ============================================================================
# CREATE DATASET
# ============================================================================

records = []

for year in [2022, 2023, 2024]:
    for company in companies:
        filing_date = f"{year}-07-15"

        record = {
            'cik': company['cik'],
            'company_name': company['name'],
            'industry': company['industry'],
            'fiscal_year': year,
            'filing_date': filing_date,
            'mda_text': generate_mda_text(company, year),
            # Metadata for validation
            'true_attitude': company['digital_attitude'],
            'true_trend': company['trend']
        }

        records.append(record)

# Create DataFrame
df = pd.DataFrame(records)

# Add some basic statistics
df['word_count'] = df['mda_text'].str.split().str.len()
df['char_count'] = df['mda_text'].str.len()

# Save to CSV
output_file = 'test_mda_dataset.csv'
df.to_csv(output_file, index=False)

print(f"✓ Generated {len(df)} MD&A sections")
print(f"  Companies: {df['company_name'].nunique()}")
print(f"  Years: {df['fiscal_year'].min()} - {df['fiscal_year'].max()}")
print(f"  Industries: {df['industry'].nunique()}")
print(f"\nAttitude Distribution:")
print(df.groupby('fiscal_year')['true_attitude'].value_counts().unstack(fill_value=0))
print(f"\nAverage word count: {df['word_count'].mean():.0f} words")
print(f"Total dataset size: {df['word_count'].sum():,} words")
print(f"\nSaved to: {output_file}")

# Show sample
print(f"\n{'='*80}")
print("SAMPLE MD&A EXCERPT (First 500 characters)")
print(f"{'='*80}")
print(f"Company: {df.iloc[0]['company_name']}")
print(f"Year: {df.iloc[0]['fiscal_year']}")
print(f"Attitude: {df.iloc[0]['true_attitude']}")
print(f"\n{df.iloc[0]['mda_text'][:500]}...")
