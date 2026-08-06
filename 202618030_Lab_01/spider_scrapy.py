import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]

    # Limit setting for listing pages
    page_count = 1
    max_pages = 10

    # Export configuration directly to CSV
    custom_settings = {
        'FEEDS': {
            'books.csv': {
                'format': 'csv',
                'overwrite': True,
            }
        },
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'CONCURRENT_REQUESTS': 8,
        'DOWNLOAD_DELAY': 0.5,  # Polite scraping delay
    }

    def parse(self, response):
        """Parse listing pages and follow links to book detail pages."""
        # Extract all book detail page links from the current page
        book_links = response.css('article.product_pod h3 a::attr(href)').getall()
        for link in book_links:
            yield response.follow(link, callback=self.parse_book_details)

        # Handle pagination up to the 5-page limit
        if self.page_count < self.max_pages:
            next_page = response.css('li.next a::attr(href)').get()
            if next_page:
                self.page_count += 1
                yield response.follow(next_page, callback=self.parse)

    def parse_book_details(self, response):
        """Parse detailed information inside each book page."""

        # Helper function to clean table data extraction
        def get_table_val(label):
            return response.xpath(
                f'//th[text()="{label}"]/following-sibling::td/text()'
            ).get()

        # Extract star rating from CSS class name (e.g., 'star-rating Three')
        rating_classes = response.css('p.star-rating::attr(class)').get()
        rating = (
            rating_classes.replace('star-rating', '').strip()
            if rating_classes
            else None
        )

        # Clean availability string
        availability_raw = response.css(
            'p.instock.availability::text'
        ).getall()
        availability = (
            "".join(availability_raw).strip() if availability_raw else None
        )

        yield {
            'title': response.css('div.product_main h1::text').get(),
            'price': response.css('p.price_color::text').get(),
            'rating': rating,
            'availability': availability,
            'category': response.css(
                'ul.breadcrumb li:nth-child(3) a::text'
            ).get(),
            'upc': get_table_val('UPC'),
            'price_excl_tax': get_table_val('Price (excl. tax)'),
            'price_incl_tax': get_table_val('Price (incl. tax)'),
            'tax': get_table_val('Tax'),
            'number_of_reviews': get_table_val('Number of reviews'),
            'description': response.css(
                '#product_description + p::text'
            ).get(),
            'url': response.url,
        }