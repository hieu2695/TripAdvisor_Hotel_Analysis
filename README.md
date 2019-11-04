# Analysis of Hotel Service in the United States

Welcome!
This project is an analysis of hotel service in the United States. 
The project performs two main processes:
1. Scraping the data from [TripvAdvisor](https://www.tripadvisor.com/) .
2. Analyzing the data.

All the codes were done using Python. To plot United States Choropleth Map, plotly was used.

---
## Scraping the data

[Notebook link](https://hieu2695.github.io/TripAdvisor_Hotel_Analysis/Scraping.html)

I used Python to scrape the data from TripAdvisor website. At the beginning, I want to scrape the data from the main page of United States hotels, but the links for different pages are them same and the "next" and "page" button have no href links to enter. Therefore, I have to change my plan. Firstly, I write a code to scrape all the links for hotel locations in the United States. After that, I can enter the websites with hotels by location using the links above. And in these websites' links, there has a variable to go to other pages. I write a code to retrieve all the links of the hotels in different pages and save them into a list of hote links. Using the for loop, I can scrape the data from each hotel link: Hotel Name, Location, State, Rating, Price, ...

In general, I have 3 steps to scrape the hotel data:
* Step 1: Retrieve the first pages of U.S locations.
* Step 2: Retrieve other pages of U.S locations by changing a variable in the first pages' url.
* Step 3: Retrieve all the hotel links from location pages and scrape the necessary data using these links.

I selected top 100 popular locations in the United States (ranked by TripAdvisor) and for each location I scraped the hotels from the first 5 pages (around 150 hotels per location).

---
## Analysis

[Notebook link](https://hieu2695.github.io/TripAdvisor_Hotel_Analysis/Hotels_Analysis.html)

Before my analysis, I need to preprocess my data. I need to fill missing locations for some hotels and change the price values from object to float. The strange values, outliers are also treated in the process. After that, I have an overview and some visualizations about the hotel service in the United States.

My main analysis focuses on the best value hotels (hotels with good ratings and reasonable prices) to figure out best places to visit in the U.S. The results can only show the places with high numbers of sightseeing options for tourists such as number of restaurants and number of attractions. They cannot figure out the best destination in the United States.

---
## Key Findings

Overall, the majority of hotels (around 83%) in the United States are rated Excellent or Very good. The average cost to stay for one night is $140. I also visualize that when the price increases, the rating seems to be better. All luxury hotels ($1000+ price/night) are rated Excellent or Very good.

After analyzing the best value hotels (rated Excellent or Very good and price is not more than $350), I find out that New York City and its state New York (NY) have the highest average numbers of attractions and restaurants nearby a hotel. The grade for walking there is also at the top, which means that visitors often feel comfortable to wander and explore the shopping malls and sightseeings in the city.
