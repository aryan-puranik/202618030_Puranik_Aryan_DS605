# Data Scraping and Preprocessing using Python and Scrapy
Name: Puranik Aryan Hitesh
ID: 202618030


# Book Store Web Scraping & Exploratory Data Analysis

A complete end-to-end data analytics project that demonstrates how to collect real-world data using **Scrapy**, preprocess it using **Python and Pandas**, and perform **Exploratory Data Analysis (EDA)** with insightful visualizations.

The project scrapes book information from **Books to Scrape**, a website built specifically for practicing web scraping, and transforms the collected data into meaningful business insights.

---

## Project Objectives

The primary objectives of this project are to:

- Scrape structured book data using the Scrapy framework.
- Extract useful attributes such as title, price, rating, category, availability, and description.
- Clean and preprocess the raw scraped dataset.
- Perform exploratory data analysis (EDA) to identify trends and patterns.
- Create informative visualizations for better interpretation.
- Demonstrate a complete data collection and analysis workflow.


## Project Structure

```
BookStore-WebScraping/
│
├── spider_scrapy.py           # Scrapy spider
├── books.csv                  # Raw scraped dataset
├── books_processed.csv        # Cleaned dataset
├── lab01.ipynb               # Data preprocessing & analysis
├── visualizations.png         # EDA plots
├── wordcloud.png             # Word Cloud visualization
├── README.md
```

##Data Collection

The dataset was collected using **Scrapy**, a powerful Python framework for web scraping.

The spider crawls the first **10 pages** of the website and extracts information from each individual book page.

### Extracted Features

- Book Title
- Price
- Rating
- Availability
- Category
- UPC
- Price (Excluding Tax)
- Price (Including Tax)
- Tax
- Number of Reviews
- Product Description
- Product URL

---

## Data Preprocessing

The raw dataset was cleaned and transformed before analysis.

### Data Cleaning Steps

- Removed unnecessary symbols from price columns.
- Converted prices to numeric values.
- Converted textual ratings into integer values (1–5).
- Extracted stock count from availability text.
- Generated description word counts.
- Created price bands (Low, Medium, High).
- Calculated a custom **Value Score**:
Value Score = Rating / Price

## Exploratory Data Analysis

The following analyses were performed:

- Distribution of Book Prices
- Distribution of Ratings
- Average Price by Category
- Price vs Rating Analysis
- Category-wise Statistics
- Missing Value Analysis
- Summary Statistics
- Word Cloud of Book Descriptions


## 🛠️ Technologies Used

- Python
- Scrapy
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- Jupyter Notebook



The notebook will:

- Clean the data
- Generate the processed dataset
- Perform EDA
- Create visualizations
- Generate the Word Cloud

---

## Dataset Source

The data was collected from:

https://books.toscrape.com/

This website is publicly available and intended for learning and practicing web scraping techniques.

---

## Future Improvements

Possible enhancements include:

- Scraping the complete website instead of limiting to 10 pages.
- Building an interactive dashboard using Streamlit or Power BI.
- Performing sentiment analysis on book descriptions.
- Developing a book recommendation system.
- Creating predictive models for book pricing.

---

## Author

**Puranik Aryan Hitesh

M.Sc. Data Science Student