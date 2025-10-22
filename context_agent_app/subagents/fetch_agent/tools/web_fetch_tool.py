import aiohttp
import ssl
import certifi
from bs4 import BeautifulSoup
import wikipediaapi

class WebFetchTool:
    def __init__(self):
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='ADK_TestApp/1.0 (https://github.com/aadi; aadi@example.com)'
)
    
    async def fetch_wikipedia(self, entity: str) -> dict:
        """Fetch Wikipedia summary (already working)."""
        try:
            page = self.wiki.page(entity)
            if page.exists():
                return {
                    "summary": page.summary[:500],
                    "url": page.fullurl,
                    "source": "Wikipedia"
                }
        except Exception as e:
            print(f"Wikipedia fetch error for {entity}: {e}")
        return {}
    
    async def fetch_google_news_rss(self, query: str, max_results: int = 5) -> list:
        """Fetch news from Google News RSS (no lxml dependency issues)."""
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl=self.ssl_context) as response:
                    xml_content = await response.text()
                    soup = BeautifulSoup(xml_content, 'html.parser')  # Use html.parser instead of xml
                    
                    articles = []
                    for item in soup.find_all('item')[:max_results]:
                        title_tag = item.find('title')
                        link_tag = item.find('link')
                        pubdate_tag = item.find('pubdate')
                        
                        articles.append({
                            'title': title_tag.text if title_tag else 'No title',
                            'link': link_tag.text if link_tag else '',
                            'published': pubdate_tag.text if pubdate_tag else '',
                            'source': 'Google News'
                        })
                    
                    return articles
        except Exception as e:
            print(f"Google News fetch error for {query}: {e}")
            return []
    
    async def fetch_multiple_entities(self, entity_names: list, include_news: bool = True) -> list:
        """Fetch context for multiple entities."""
        results = []
        
        for entity in entity_names:
            context = {
                "entity": entity,
                "wikipedia": await self.fetch_wikipedia(entity),
                "news": []
            }
            
            if include_news:
                context["news"] = await self.fetch_google_news_rss(entity, max_results=3)
            
            results.append(context)
        
        return results
    
    def format_context_for_llm(self, context_data: list) -> str:
        """Format fetched context into readable text."""
        formatted = []
        
        for item in context_data:
            entity = item.get("entity", "Unknown")
            formatted.append(f"\n=== {entity} ===")
            
            # Wikipedia
            wiki = item.get("wikipedia", {})
            if wiki:
                formatted.append(f"📚 Wikipedia: {wiki.get('summary', 'No summary')[:200]}...")
            
            # News
            news = item.get("news", [])
            if news:
                formatted.append(f"📰 Recent News ({len(news)} articles):")
                for article in news[:3]:
                    formatted.append(f"  • {article.get('title', 'No title')}")
        
        return "\n".join(formatted)

# Create singleton instance
web_fetch_tool = WebFetchTool()
