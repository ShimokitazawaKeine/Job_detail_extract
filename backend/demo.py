# Set up
import pandas as pd
import numpy as np
import re
import pickle
import os
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


import string

# nltk
class BasicSkillExtractor:
    """Try to extract all skills mentioned in job descriptions"""
    
    def __init__(self):
        # skill keywords library
        self.skill_keywords = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
                'go', 'rust', 'scala', 'kotlin', 'swift', 'typescript', 'sql',
                'r', 'matlab', 'perl', 'shell', 'bash', 'powershell', 'vba'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'google cloud', 'kubernetes', 'docker',
                'terraform', 'ansible', 'jenkins', 'devops', 'cloud formation',
                'helm', 'openshift', 'vagrant', 'puppet', 'chef'
            ],
            'security': [
                'cybersecurity', 'penetration testing', 'vulnerability',
                'firewall', 'encryption', 'siem', 'cissp', 'cism', 'cisa',
                'security operations', 'incident response', 'malware analysis',
                'threat hunting', 'forensics', 'compliance', 'audit'
            ],
            'database': [
                'mysql', 'postgresql', 'mongodb', 'oracle', 'redis',
                'elasticsearch', 'sql server', 'database', 'cassandra',
                'dynamodb', 'sqlite', 'mariadb', 'nosql', 'neo4j'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'spring',
                'nodejs', 'express', 'laravel', 'rails', 'asp.net',
                'dotnet', 'bootstrap', 'jquery', 'hibernate'
            ],
            'tools': [
                'git', 'jira', 'confluence', 'splunk', 'grafana', 'nagios',
                'linux', 'windows', 'unix', 'powershell', 'bash', 'gitlab',
                'github', 'bitbucket', 'slack', 'teams', 'zoom'
            ],
            'soft_skills': [
                'communication', 'verbal communication', 'written communication',
                'presentation skills', 'public speaking', 'interpersonal skills',
                'listening skills', 'negotiation', 'persuasion',
                'leadership', 'team leadership', 'management', 'project management',
                'people management', 'mentoring', 'coaching', 'delegation',
                'decision making', 'strategic thinking',
                'teamwork', 'collaboration', 'team player', 'cross-functional',
                'stakeholder management', 'relationship building', 'networking',
                'problem solving', 'analytical thinking', 'critical thinking',
                'troubleshooting', 'debugging', 'root cause analysis',
                'logical thinking', 'attention to detail',
                'adaptability', 'flexibility', 'learning agility', 'continuous learning',
                'quick learner', 'self-motivated', 'proactive', 'innovative',
                'creative thinking', 'open minded',
                'work ethic', 'reliability', 'punctuality', 'accountability',
                'responsibility', 'integrity', 'professional', 'ethical',
                'commitment', 'dedication',
                'time management', 'organization', 'planning', 'prioritization',
                'multitasking', 'efficiency', 'productivity', 'goal oriented',
                'deadline management', 'scheduling',
                'customer service', 'client facing', 'customer focus',
                'service oriented', 'customer satisfaction', 'support',
                'stress management', 'emotional intelligence', 'resilience',
                'patience', 'empathy', 'conflict resolution', 'composure',
                'innovation', 'creativity', 'thinking outside the box',
                'brainstorming', 'ideation', 'design thinking'
            ],
            'business_skills': [
                'business analysis', 'requirements gathering', 'process improvement',
                'workflow optimization', 'business intelligence', 'data analysis',
                'project management', 'agile', 'scrum', 'kanban', 'waterfall',
                'pmp', 'prince2', 'risk management', 'budget management',
                'sales', 'marketing', 'business development', 'account management',
                'relationship management', 'revenue generation',
                'budgeting', 'financial analysis', 'cost management',
                'roi analysis', 'financial planning'
            ],
            'industry_knowledge': [
                'finance', 'healthcare', 'education', 'retail', 'manufacturing',
                'telecommunications', 'government', 'non-profit', 'consulting',
                'compliance', 'regulatory', 'audit', 'governance', 'policy',
                'sox', 'gdpr', 'hipaa', 'pci dss'
            ]
        }

        # Soft skill synonyms mapping for better recognition
        self.soft_skill_synonyms = {
            'communication': ['communicate', 'communicating', 'verbal', 'written'],
            'leadership': ['lead', 'leading', 'leader', 'manage', 'managing', 'manager'],
            'teamwork': ['team work', 'collaborate', 'collaboration', 'collaborative', 'cooperative'],
            'problem solving': ['problem-solving', 'troubleshoot', 'debug', 'resolve', 'solving'],
            'adaptability': ['adaptable', 'flexible', 'adjust', 'versatile'],
            'time management': ['time-management', 'organize', 'organizing', 'planning', 'schedule'],
            'customer service': ['customer-service', 'client service', 'support', 'customer support'],
            'analytical thinking': ['analytical', 'analyze', 'analysis', 'analytical skills'],
            'project management': ['project-management', 'managing projects', 'pm'],
            'strategic thinking': ['strategic', 'strategy', 'strategic planning']
        }

        # Additional patterns for skill extraction
        self.skill_patterns = [
            r'experience\s+(?:in|with)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'skilled\s+(?:in|with)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'knowledge\s+(?:of|in)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'proficient\s+(?:in|with)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'expertise\s+(?:in|with)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'familiar\s+(?:with)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'understanding\s+(?:of)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'background\s+(?:in|with)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
            r'working\s+(?:with|in)\s+([a-zA-Z0-9\s\-\.]+?)(?:[,;.]|$|\sand\s)',
        ]

        # Technical abbreviations and acronyms
        self.tech_patterns = [
            r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',  # All caps abbreviations
            r'\b\w+(?:\.\w+)+\b',  # Tech terms like asp.net, node.js
            r'\b\w+\+\+?\b',  # Languages like C++
            r'\b\w+#\b'  # Languages like C#
        ]
    
    def extract_all_skills(self, text):
        """
        Extract ALL skills mentioned in the job description
        Returns a complete list of skill keywords
        """
        if not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        all_skills = set()  # Use set to avoid duplicates
        
        # Method 1: Direct keyword matching
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    all_skills.add(keyword)
        
        # Method 2: Handle synonyms (especially for soft skills)
        for main_skill, synonyms in self.soft_skill_synonyms.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    all_skills.add(main_skill)
        
        # Method 3: Pattern-based extraction
        for pattern in self.skill_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned_skills = self._clean_extracted_text(match)
                all_skills.update(cleaned_skills)
        
        # Method 4: Technical terms and abbreviations
        for pattern in self.tech_patterns:
            matches = re.findall(pattern, text)  # Use original case for abbreviations
            for match in matches:
                if self._is_valid_tech_term(match):
                    all_skills.add(match.lower())
        
        # Convert to sorted list for consistent output
        return sorted(list(all_skills))
    
    def _clean_extracted_text(self, text):
        """Clean and split extracted text into individual skills"""
        if not text or len(text.strip()) < 2:
            return []
        
        # Common noise words to filter out
        noise_words = {
            'experience', 'knowledge', 'skills', 'ability', 'working', 'strong', 
            'good', 'excellent', 'proven', 'demonstrated', 'years', 'minimum', 
            'required', 'preferred', 'plus', 'bonus', 'understanding', 'background',
            'familiarity', 'exposure', 'hands', 'on', 'hands-on'
        }
        
        skills = []
        # Split by common delimiters
        parts = re.split(r'[,;&]|\sand\s|\sor\s', text.strip())
        
        for part in parts:
            part = part.strip()
            if len(part) > 1 and len(part) < 50:  # Reasonable length
                # Remove noise words
                words = part.split()
                clean_words = [w for w in words if w.lower() not in noise_words]
                
                if clean_words:
                    cleaned_skill = ' '.join(clean_words)
                    if len(cleaned_skill) > 1:
                        skills.append(cleaned_skill)
        
        return skills
    
    def _is_valid_tech_term(self, term):
        """Check if a term is a valid technical term"""
        # Filter out common non-technical abbreviations
        invalid_terms = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 
            'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HAVE', 'HIS', 'HAS',
            'TO', 'OF', 'IN', 'IT', 'ON', 'BE', 'AT', 'BY', 'WE', 'IS'
        }
        
        if term.upper() in invalid_terms:
            return False
        
        # Must be reasonable length
        if len(term) < 2 or len(term) > 20:
            return False
        
        return True
    
    def extract_skills_by_category(self, text):
        """
        Extract skills and group them by category
        Returns dictionary with categories as keys and skill lists as values
        """
        if not isinstance(text, str):
            return {}
        
        text_lower = text.lower()
        categorized_skills = {}
        
        # Extract skills by predefined categories
        for category, keywords in self.skill_keywords.items():
            found_skills = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    found_skills.append(keyword)
            
            # Handle synonyms for soft skills
            if category == 'soft_skills':
                for main_skill, synonyms in self.soft_skill_synonyms.items():
                    for synonym in synonyms:
                        if synonym in text_lower and main_skill not in found_skills:
                            found_skills.append(main_skill)
            
            if found_skills:
                categorized_skills[category] = sorted(list(set(found_skills)))
        
        # Extract additional skills using patterns
        pattern_skills = []
        for pattern in self.skill_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                cleaned_skills = self._clean_extracted_text(match)
                pattern_skills.extend(cleaned_skills)
        
        if pattern_skills:
            # Remove duplicates from pattern skills that are already categorized
            all_categorized = set()
            for skills in categorized_skills.values():
                all_categorized.update(skills)
            
            unique_pattern_skills = [skill for skill in pattern_skills 
                                   if skill not in all_categorized]
            
            if unique_pattern_skills:
                categorized_skills['other_skills'] = sorted(list(set(unique_pattern_skills)))
        
        return categorized_skills
    
    def get_skill_summary(self, text):
        """
        Get a comprehensive summary of all extracted skills
        """
        all_skills = self.extract_all_skills(text)
        categorized_skills = self.extract_skills_by_category(text)
        
        total_skills = len(all_skills)
        category_counts = {cat: len(skills) for cat, skills in categorized_skills.items()}
        
        return {
            'all_skills': all_skills,
            'total_count': total_skills,
            'categorized_skills': categorized_skills,
            'category_counts': category_counts,
            'skill_density': total_skills / len(text.split()) if text else 0
        }

