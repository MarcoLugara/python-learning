"""
Sample data for video3_advanced_prompts_demo.py
"""

# Sales data for complex nested parsing example
sales_data = """
Q4 2024 Sales Report:
- Electronics: $2.4M (up 15% YoY)
- Clothing: $1.8M (down 5% YoY)
- Home Goods: $1.2M (up 8% YoY)
- Total Revenue: $5.4M (up 6% YoY)

Notable trends:
- Black Friday sales exceeded projections by 22%
- Online sales grew 35%, now 60% of total
- Customer retention rate: 78%
- Average order value: $145 (up from $132 last year)
"""

# Customer feedback samples for few-shot prompting
customer_feedback_samples = [
    "App crashes on submit",
    "Great customer service!",
    "The new dashboard is confusing and hard to navigate",
    "Payment processing is too slow",
    "Love the new features!",
    "Unable to export reports in CSV format"
]

# Market data for chain-of-thought prompting
market_data_cot = """
Market Overview (Q4 2024):
- Industry growth rate: 12% annually
- Our company growth: 6% annually
- Top 3 competitors average: 15% growth
- Market share: We have 8%, down from 10% last year
- Customer acquisition cost: $85 (industry avg: $65)
- Customer lifetime value: $450 (industry avg: $380)

Recent developments:
- Competitor A launched new mobile app (Sept 2024)
- Pricing pressure in electronics segment
- Supply chain costs decreased 8%
"""

# Market data for constrained analysis
market_data_constrained = """
Stock Performance Data (Last Quarter):
- Tech Sector Index: +8.5%
- Our Portfolio Holdings:
  * TechCorp: +12.3%
  * DataSystems: -2.1%
  * CloudServices: +15.7%
  * AIInnovate: +6.8%

Trading volumes:
- TechCorp: 2.3M shares/day avg
- DataSystems: 1.1M shares/day avg
- CloudServices: 3.5M shares/day avg
- AIInnovate: 1.8M shares/day avg

Note: All data is historical. Past performance does not guarantee future results.
"""