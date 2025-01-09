# app/core/scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from pathlib import Path
from .config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CourseScraper:
    def __init__(self, base_url="https://courses.analyticsvidhya.com/courses"):
        self.base_url = base_url
        self.raw_data_path = Path(settings.RAW_DATA_PATH)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
    def fetch_courses(self):
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            return self._parse_courses(response.content)
        except Exception as e:
            logger.error(f"Error fetching courses: {e}")
            return None

    def _parse_courses(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        courses = []
        
        for course in soup.find_all('div', class_='course-card'):
            try:
                course_data = {
                    'title': course.find('h2').text.strip(),
                    'description': course.find('p', class_='description').text.strip(),
                    'level': course.find('span', class_='level').text.strip(),
                    'duration': course.find('span', class_='duration').text.strip(),
                    'url': course.find('a')['href'],
                    'instructor': course.find('span', class_='instructor').text.strip(),
                    'rating': course.find('span', class_='rating').text.strip(),
                }
                courses.append(course_data)
            except AttributeError as e:
                logger.warning(f"Error parsing course: {e}")
                continue
                
        return courses

    def save_courses(self, courses):
        if not courses:
            return None
            
        df = pd.DataFrame(courses)
        output_path = self.raw_data_path / 'courses.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(courses)} courses to {output_path}")
        return df

    def run(self):
        courses = self.fetch_courses()
        return self.save_courses(courses)