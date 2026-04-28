# REI Survey Intelligence Dashboard

## Description

An interactive data dashboard designed to help NGOs analyze survey data, identify community needs, and make data-driven decisions.

The dashboard turns raw survey responses from beneficiaries, volunteers, or outreach participants into clean analytics, practical insights, vulnerability scores, and downloadable reports.

## Features

- Data upload for CSV and Excel files
- Interactive filters for gender, location, program interest, and education level
- Real-time analytics that update when filters change
- Demographic and needs-assessment charts
- Automated key findings and interpretation
- AI-style recommendation engine
- Vulnerability scoring system to identify people who may need urgent help
- Donor-ready PDF report download
- Cleaned CSV, Excel, text report, PDF report, and vulnerability score downloads

## Tech Stack

- Python
- Pandas
- Streamlit
- Matplotlib
- OpenPyXL

## Use Case

Built for NGOs like Rofiyat Empowerment Initiative to improve decision-making, understand community needs, identify vulnerable beneficiaries, and allocate resources more effectively.

## Dashboard Pages

### Overview

- Total filtered responses
- Most common location
- Most common challenge
- Average satisfaction score
- Location, program demand, and satisfaction charts

### Demographics

- Gender distribution
- Age group breakdown
- Location breakdown
- Education level breakdown

### Needs Assessment

- Main challenges
- Program interests
- Digital skill levels
- Financial literacy levels
- Health awareness levels

### Insights

- Key findings
- Interpretation of what the data means
- Recommendations for next NGO actions
- Vulnerability scoring system
- Urgent-help beneficiary flags
- AI-style recommendation engine
- Donor PDF report download

## Data Columns

The app works best with these survey columns:

| Column | Description |
| --- | --- |
| Name | Respondent name or identifier |
| Age | Respondent age |
| Gender | Gender category |
| Location | Community, city, state, or project location |
| Education Level | Highest education level |
| Employment Status | Current employment status |
| Main Challenge | Biggest challenge selected by the respondent |
| Program Interest | Program or support area the respondent wants |
| Digital Skill Level | Self-reported digital skill level |
| Health Awareness Level | Self-reported health awareness level |
| Financial Literacy Level | Self-reported financial literacy level |
| Satisfaction Score | Program satisfaction score, ideally from 1 to 5 |
| Recommendation | Open-ended feedback or recommendation |

If a file is missing one of these columns, the app fills the missing field with `Not provided` so the dashboard can still run.

## Project Structure

```bash
survey-intelligence-dashboard/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- sample_data.csv
|-- assets/
`-- .gitignore
```

The `assets/` folder is optional and can be added later for images, logos, or screenshots.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```bash
http://localhost:8501
```

## How to Use

1. Start the Streamlit app.
2. Upload a CSV or Excel survey file, or use the included sample data.
3. Select filters in the sidebar.
4. Review the Overview, Demographics, Needs Assessment, and Insights tabs.
5. Download cleaned data, vulnerability scores, or donor-ready reports.

## Deployment

To deploy on Streamlit Cloud:

1. Push this project to GitHub.
2. Go to Streamlit Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

## Portfolio Summary

This project demonstrates how Python, Pandas, Streamlit, and Matplotlib can be used to build a practical NGO data product. It supports data cleaning, interactive filtering, real-time analytics, automated insights, vulnerability scoring, and report generation for better program planning and donor communication.
