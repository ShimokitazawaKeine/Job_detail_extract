#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Job Classification and skillset extraction Web Application: Flask Backend
Optimized for Google Cloud deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置CORS - 适配生产环境
if os.getenv('ENVIRONMENT') == 'production':
    # 生产环境 - 允许来自Cloud Storage的前端访问
    CORS(app, origins=[
        'https://storage.googleapis.com',
        'https://storage.cloud.google.com',
        # 如果你有自定义域名，在这里添加
    ])
else:
    # 开发环境 - 允许本地访问
    CORS(app, origins=['http://localhost:3000'])

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# 在云环境中不需要本地文件存储
if os.getenv('ENVIRONMENT') != 'production':
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
classifier = None
model_loaded = False
skill_extractor = None
extractor_mode = None

# Job category mapping (与你原有的保持一致)
JOB_CATEGORIES = {
    0: {
        "name": "CyberSecurity Consultant",
        "description": "Cybersecurity consulting and advisory roles",
        "color": "#E91E63",
    },
    1: {
        "name": "CyberSecurity Analyst", 
        "description": "Security analysis and monitoring positions",
        "color": "#2196F3",
    },
    2: {
        "name": "CyberSecurity Architect",
        "description": "Security architecture and design roles",
        "color": "#9C27B0",
    },
    3: {
        "name": "CyberSecurity Operations",
        "description": "Security operations and incident response",
        "color": "#FF5722",
    },
    4: {
        "name": "Information Security",
        "description": "Information security management and governance",
        "color": "#607D8B",
    },
    5: {
        "name": "CyberSecurity Testers",
        "description": "Penetration testing and security assessment",
        "color": "#FF9800",
    }
}

# 导入你的自定义模块
try:
    from classifier import JobClassifier
    from model import HybridSkillExtractor
    from demo import BasicSkillExtractor
except ImportError as e:
    logger.warning(f"Failed to import custom modules: {e}")
    # 创建基础的提取器类
    class BasicSkillExtractor:
        def extract_all_skills(self, text):
            # 简化的技能提取逻辑
            basic_skills = [
                'python', 'java', 'javascript', 'cybersecurity', 'aws', 'azure',
                'docker', 'kubernetes', 'mysql', 'react', 'angular', 'git'
            ]
            found_skills = []
            text_lower = text.lower()
            for skill in basic_skills:
                if skill in text_lower:
                    found_skills.append(skill)
            return found_skills

def load_model():
    """加载模型和技能提取器"""
    global classifier, model_loaded, skill_extractor, extractor_mode

    try:
        # 尝试加载训练好的模型
        model_path = 'job_classifier_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            model_loaded = True
            logger.info("Trained model loaded successfully.")
        else:
            logger.warning("Model file not found. Using demo mode.")
            model_loaded = False

        # 尝试加载混合技能提取器，失败则使用基础版本
        try:
            from model import HybridSkillExtractor
            skill_extractor = HybridSkillExtractor()
            extractor_mode = 'hybrid'
            logger.info("HybridSkillExtractor loaded.")
        except Exception as e:
            skill_extractor = BasicSkillExtractor()
            extractor_mode = 'basic'
            logger.warning(f"Using BasicSkillExtractor. Reason: {e}")

        return True

    except Exception as e:
        logger.error(f"Failed to load model or extractor: {e}")
        # 确保至少有基础提取器
        skill_extractor = BasicSkillExtractor()
        extractor_mode = 'basic'
        return False

def classify_job_category(text):
    """规则基础的职位分类"""
    text = text.lower()

    category_keywords = {
        1: ['analyst', 'monitoring', 'detection', 'incident analyst', 'blue team'],
        2: ['architect', 'architecture', 'design security', 'security by design'],
        3: ['devops', 'operations', 'infrastructure', 'system admin', 'cloud ops'],
        4: ['governance', 'compliance', 'risk management', 'grc', 'ciso', 'policy'],
        5: ['penetration testing', 'pentester', 'ethical hacker', 'red team', 'vulnerability'],
        0: ['consultant', 'advisory', 'solution consultant', 'security consultant'],
    }

    for category_id, keywords in category_keywords.items():
        if any(kw in text for kw in keywords):
            return category_id

    return 0  # 默认: Consultant

def demo_predict(title, skillset):
    """演示模式的预测逻辑"""
    text = f"{title} {skillset}"
    category = classify_job_category(text)

    try:
        if extractor_mode == 'hybrid':
            extraction = skill_extractor.extract_skills(skillset)
            skills = extraction.get("combined_skills", [])
        else:
            skills = skill_extractor.extract_all_skills(skillset)
    except Exception as e:
        logger.warning(f"Skill extraction failed: {e}")
        skills = []

    return {
        'job_category': category,
        'skills': skills
    }

# API 路由

@app.route('/api/predict', methods=['POST'])
def predict_job():
    """单个工作预测API - 匹配前端期望的响应格式"""
    try:
        data = request.get_json()
        
        if not data or 'title' not in data or 'skillset' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: title and skillset'
            }), 400

        title = data['title'].strip()
        skillset = data['skillset'].strip()

        if not title or not skillset:
            return jsonify({
                'success': False,
                'error': 'Title and skillset cannot be empty'
            }), 400

        # 1. 职位分类
        if model_loaded and classifier:
            result = classifier.predict(title, skillset)
        else:
            result = demo_predict(title, skillset)

        # 2. 技能提取
        try:
            if extractor_mode == 'hybrid':
                extraction = skill_extractor.extract_skills(skillset)
                skills = extraction.get("combined_skills", [])
            else:
                skills = skill_extractor.extract_all_skills(skillset)
        except Exception as e:
            logger.warning(f"Skill extraction failed: {e}")
            skills = []

        # 3. 组合响应 - 匹配前端期望的格式
        category_id = result['job_category']
        category_info = JOB_CATEGORIES.get(category_id, {
            "name": f"Category {category_id}",
            "description": "Unknown category",
            "color": "#757575",
        })

        return jsonify({
            'success': True,
            'prediction': {
                'category_id': category_id,
                'category_name': category_info['name'],
                'category_description': category_info['description'],
                'category_color': category_info['color']
            },
            'skills': {
                'extracted_skills': skills[:20],  # 限制返回数量，避免响应过大
                'skill_count': len(skills)
            },
            'model_info': {
                'mode': 'trained_model' if model_loaded else 'demo_mode',
                'extractor_mode': extractor_mode,
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Error during prediction: {str(e)}'
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """批量预测API - 匹配前端期望的响应格式"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False, 
                'error': 'Please upload a CSV file'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False, 
                'error': 'No file selected'
            }), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                'success': False, 
                'error': 'Only CSV files are supported'
            }), 400

        # 直接读取文件内容，不保存到磁盘（适合云环境）
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read CSV file: {str(e)}'
            }), 400

        # 验证必需的列
        required_columns = ['title', 'skillset']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'success': False, 
                'error': f'Missing columns: {", ".join(missing_columns)}'
            }), 400

        results = []
        processed_count = 0

        for index, row in df.iterrows():
            try:
                title = str(row['title']).strip()
                skillset = str(row['skillset']).strip()

                if not title or not skillset or title == 'nan' or skillset == 'nan':
                    continue

                # 使用相同的预测逻辑
                if model_loaded and classifier:
                    result = classifier.predict(title, skillset)
                else:
                    result = demo_predict(title, skillset)

                category_info = JOB_CATEGORIES.get(result['job_category'], {})

                # 提取技能
                try:
                    if extractor_mode == 'hybrid':
                        extraction = skill_extractor.extract_skills(skillset)
                        skills = extraction.get("combined_skills", [])
                    else:
                        skills = skill_extractor.extract_all_skills(skillset)
                except:
                    skills = []

                results.append({
                    'row_index': index,
                    'title': title,
                    'predicted_category': category_info.get('name', f"Category {result['job_category']}"),
                    'skills_found': len(skills),
                    'skills': ", ".join(skills[:20])  # 限制技能数量
                })
                
                processed_count += 1
                
                # 限制处理数量，避免超时
                if processed_count >= 2000:
                    break

            except Exception as e:
                logger.warning(f"Error processing row {index}: {e}")
                continue

        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_jobs': len(results),
                'categories_found': len(set(r['predicted_category'] for r in results)),
                'model_mode': 'trained_model' if model_loaded else 'demo_mode',
                'extractor_mode': extractor_mode
            }
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取应用状态"""
    return jsonify({
        'success': True,
        'status': {
            'model_loaded': model_loaded,
            'extractor_mode': extractor_mode,
            'categories_count': len(JOB_CATEGORIES),
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'mode': 'trained_model' if model_loaded else 'demo_mode'
        }
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """获取所有工作类别"""
    return jsonify({
        'success': True,
        'categories': JOB_CATEGORIES
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'extractor_mode': extractor_mode
    })

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Requested resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

# 应用启动
if __name__ == '__main__':
    print("Starting job classification web application...")
    print("=" * 50)
    
    # 加载模型
    load_model()
    
    if model_loaded:
        print("✅ Model loaded: using trained model")
    else:
        print("⚠️  Model not loaded: using demo mode")
    
    print(f"📊 Supported job categories: {len(JOB_CATEGORIES)}")
    print(f"🔧 Skill extractor mode: {extractor_mode}")
    
    # 根据环境选择启动方式
    if os.getenv('ENVIRONMENT') == 'production':
        print("🚀 Starting production server...")
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("🛠️  Starting development server...")
        print("🌐 Access: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    
    print("=" * 50)