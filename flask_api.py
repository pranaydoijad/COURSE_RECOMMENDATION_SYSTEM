"""
Course Recommendation System - Flask REST API
Phase 3: Deployment

Run with: python flask_api.py
API will be available at http://localhost:5000

Example requests:
GET http://localhost:5000/api/recommend?user_id=123&algorithm=hybrid&k=5
GET http://localhost:5000/api/user/123
GET http://localhost:5000/api/courses
POST http://localhost:5000/api/batch_recommend (with JSON body)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# RECOMMENDER CLASS DEFINITIONS
# ============================================================================

class PopularityRecommender:
    """Recommend most popular courses user hasn't taken"""

    def __init__(self, course_summary):
        self.course_summary = course_summary.sort_values(
            'enrollment_numbers',
            ascending=False
        )
        self.all_courses = set(course_summary.index)

    def recommend(self, user_id, user_history, k=5):
        available_courses = self.all_courses - user_history
        if len(available_courses) == 0:
            return []

        recommendations = self.course_summary.loc[
            self.course_summary.index.isin(available_courses)
        ].head(k)

        return recommendations.index.tolist()


class ContentBasedRecommender:
    """Recommend courses similar to user's history"""

    def __init__(self, similarity_matrix, course_summary):
        self.similarity_matrix = similarity_matrix
        self.course_summary = course_summary
        self.all_courses = set(course_summary.index)

    def recommend(self, user_id, user_history, k=5):
        if len(user_history) == 0:
            return self.course_summary.nlargest(k, 'enrollment_numbers').index.tolist()

        similarity_scores = {}
        for course in self.all_courses:
            if course in user_history:
                similarity_scores[course] = -1
            else:
                similarities = [
                    self.similarity_matrix.loc[course, taken_course]
                    for taken_course in user_history
                    if taken_course in self.similarity_matrix.index
                ]
                similarity_scores[course] = np.mean(similarities) if similarities else 0

        recommendations = sorted(
            similarity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [course for course, score in recommendations[:k]]


class HybridRecommender:
    """Blend popularity and content-based recommendations"""

    def __init__(self, pop_recommender, content_recommender, course_summary, popularity_weight=0.6):
        self.pop_recommender = pop_recommender
        self.content_recommender = content_recommender
        self.course_summary = course_summary
        self.popularity_weight = popularity_weight
        self.content_weight = 1 - popularity_weight
        self.all_courses = set(course_summary.index)

    def recommend(self, user_id, user_history, k=5):
        pop_recs = self.pop_recommender.recommend(user_id, user_history, k=20)
        content_recs = self.content_recommender.recommend(user_id, user_history, k=20)

        scores = {}
        for rank, course in enumerate(pop_recs):
            pop_score = 1 - (rank / len(pop_recs))
            scores[course] = scores.get(course, 0) + self.popularity_weight * pop_score

        for rank, course in enumerate(content_recs):
            content_score = 1 - (rank / len(content_recs))
            scores[course] = scores.get(course, 0) + self.content_weight * content_score

        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [course for course, score in recommendations[:k]]


# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# LOAD MODELS
# ============================================================================

try:
    with open('recommender_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    pop_recommender = models['popularity']
    content_recommender = models['content_based']
    hybrid_recommender = models['hybrid']
    course_summary = models['course_summary']
    metrics = models['metrics']
    
    # Load training data
    df_train = pd.read_excel('online_course_recommendation_v2.xlsx')
    df_train['completed'] = (df_train['time_spent_hours'] >= df_train['course_duration_hours'] * 0.5).astype(int)
    df_train['high_satisfaction'] = (df_train['rating'] >= 4).astype(int)
    
    print("✅ Models loaded successfully")

except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("Please run the Complete_Course_Recommendation_System.ipynb first")
    exit(1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_user_history(user_id):
    """Get user's course history"""
    user_data = df_train[df_train['user_id'] == user_id]
    if user_data.empty:
        return None
    
    return {
        'user_id': int(user_id),
        'courses_taken': user_data['course_name'].unique().tolist(),
        'num_courses': len(user_data['course_name'].unique()),
        'avg_rating': float(user_data['rating'].mean()),
        'completion_rate': float(user_data['completed'].mean()),
        'satisfaction_rate': float(user_data['high_satisfaction'].mean()),
        'total_enrollments': len(user_data)
    }

def get_recommendations(user_id, algorithm='hybrid', k=5):
    """Get recommendations for a user"""
    user_history = df_train[df_train['user_id'] == user_id]['course_name'].unique()
    user_history_set = set(user_history)
    
    # Select recommender
    if algorithm == 'popularity':
        recommender = pop_recommender
    elif algorithm == 'content_based':
        recommender = content_recommender
    elif algorithm == 'hybrid':
        recommender = hybrid_recommender
    else:
        return None
    
    # Get recommendations
    recommendations = recommender.recommend(user_id, user_history_set, k=k)
    
    # Enrich with course details
    result = []
    for i, course_name in enumerate(recommendations, 1):
        if course_name in course_summary.index:
            course_info = course_summary.loc[course_name]
            result.append({
                'rank': i,
                'course_name': course_name,
                'instructor': course_info['instructor'],
                'rating': float(course_info['rating']),
                'difficulty': course_info['difficulty_level'],
                'price': float(course_info['course_price']),
                'duration_hours': float(course_info['course_duration_hours']),
                'certification': course_info['certification_offered'],
                'enrollments': int(course_info['enrollment_numbers']),
                'feedback_score': float(course_info['feedback_score'])
            })
    
    return result

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Course Recommendation System API'
    }), 200

# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@app.route('/api/recommend', methods=['GET'])
def recommend():
    """
    Get recommendations for a user
    
    Query parameters:
        - user_id (required): User ID
        - algorithm (optional): 'hybrid', 'popularity', 'content_based' (default: hybrid)
        - k (optional): Number of recommendations (default: 5)
    
    Example:
        GET /api/recommend?user_id=123&algorithm=hybrid&k=5
    """
    try:
        user_id = request.args.get('user_id', type=int)
        algorithm = request.args.get('algorithm', default='hybrid', type=str)
        k = request.args.get('k', default=5, type=int)
        
        # Validate inputs
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        if algorithm not in ['hybrid', 'popularity', 'content_based']:
            return jsonify({'error': 'algorithm must be: hybrid, popularity, or content_based'}), 400
        
        if k < 1 or k > 20:
            return jsonify({'error': 'k must be between 1 and 20'}), 400
        
        # Get recommendations
        recommendations = get_recommendations(user_id, algorithm, k)
        
        if recommendations is None:
            return jsonify({'error': 'Invalid algorithm'}), 400
        
        return jsonify({
            'user_id': user_id,
            'algorithm': algorithm,
            'num_recommendations': len(recommendations),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """
    Get user profile and history
    
    Example:
        GET /api/user/123
    """
    try:
        user_data = get_user_history(user_id)
        
        if user_data is None:
            return jsonify({'error': f'User {user_id} not found'}), 404
        
        return jsonify({
            'user': user_data,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# COURSE ENDPOINTS
# ============================================================================

@app.route('/api/courses', methods=['GET'])
def get_courses():
    """
    Get all courses in the catalog
    
    Query parameters:
        - difficulty (optional): 'Beginner', 'Intermediate', 'Advanced'
        - sort_by (optional): 'enrollment', 'rating', 'price', 'duration'
    
    Example:
        GET /api/courses?difficulty=Beginner&sort_by=rating
    """
    try:
        difficulty = request.args.get('difficulty', type=str)
        sort_by = request.args.get('sort_by', default='enrollment', type=str)
        
        # Filter by difficulty
        courses = course_summary.copy()
        if difficulty:
            if difficulty not in ['Beginner', 'Intermediate', 'Advanced']:
                return jsonify({'error': 'difficulty must be: Beginner, Intermediate, or Advanced'}), 400
            courses = courses[courses['difficulty_level'] == difficulty]
        
        # Sort
        if sort_by == 'enrollment':
            courses = courses.sort_values('enrollment_numbers', ascending=False)
        elif sort_by == 'rating':
            courses = courses.sort_values('rating', ascending=False)
        elif sort_by == 'price':
            courses = courses.sort_values('course_price', ascending=True)
        elif sort_by == 'duration':
            courses = courses.sort_values('course_duration_hours', ascending=True)
        else:
            return jsonify({'error': 'sort_by must be: enrollment, rating, price, or duration'}), 400
        
        # Format response
        courses_list = []
        for course_name, course_data in courses.iterrows():
            courses_list.append({
                'course_name': course_name,
                'instructor': course_data['instructor'],
                'difficulty': course_data['difficulty_level'],
                'rating': float(course_data['rating']),
                'price': float(course_data['course_price']),
                'duration_hours': float(course_data['course_duration_hours']),
                'certification': course_data['certification_offered'],
                'enrollments': int(course_data['enrollment_numbers']),
                'feedback_score': float(course_data['feedback_score'])
            })
        
        return jsonify({
            'total_courses': len(courses_list),
            'courses': courses_list,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/courses/<course_name>', methods=['GET'])
def get_course_details(course_name):
    """
    Get details for a specific course
    
    Example:
        GET /api/courses/Python%20for%20Beginners
    """
    try:
        if course_name not in course_summary.index:
            return jsonify({'error': f'Course "{course_name}" not found'}), 404
        
        course_data = course_summary.loc[course_name]
        
        return jsonify({
            'course': {
                'course_name': course_name,
                'instructor': course_data['instructor'],
                'difficulty': course_data['difficulty_level'],
                'rating': float(course_data['rating']),
                'price': float(course_data['course_price']),
                'duration_hours': float(course_data['course_duration_hours']),
                'certification': course_data['certification_offered'],
                'enrollments': int(course_data['enrollment_numbers']),
                'feedback_score': float(course_data['feedback_score']),
                'completion_rate': float(course_data['completed']),
                'total_users': int(course_data['total_users'])
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# BATCH ENDPOINTS
# ============================================================================

@app.route('/api/batch_recommend', methods=['POST'])
def batch_recommend():
    """
    Get recommendations for multiple users
    
    POST body (JSON):
    {
        "users": [
            {"user_id": 123, "algorithm": "hybrid", "k": 5},
            {"user_id": 456, "algorithm": "popularity", "k": 3}
        ]
    }
    
    Example:
        POST /api/batch_recommend
        Content-Type: application/json
        {"users": [{"user_id": 123}, {"user_id": 456}]}
    """
    try:
        data = request.get_json()
        
        if not data or 'users' not in data:
            return jsonify({'error': 'Request body must contain "users" array'}), 400
        
        users = data['users']
        results = []
        
        for user_request in users:
            user_id = user_request.get('user_id')
            algorithm = user_request.get('algorithm', 'hybrid')
            k = user_request.get('k', 5)
            
            if not user_id:
                results.append({'error': 'user_id required', 'status': 'failed'})
                continue
            
            recommendations = get_recommendations(user_id, algorithm, k)
            results.append({
                'user_id': user_id,
                'algorithm': algorithm,
                'recommendations': recommendations,
                'status': 'success'
            })
        
        return jsonify({
            'batch_size': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get system performance metrics
    
    Example:
        GET /api/metrics
    """
    try:
        return jsonify({
            'popularity': {
                'precision_at_5': metrics['popularity']['precision_at_k'],
                'coverage': metrics['popularity']['coverage'],
                'num_users_evaluated': metrics['popularity']['num_users']
            },
            'content_based': {
                'precision_at_5': metrics['content_based']['precision_at_k'],
                'coverage': metrics['content_based']['coverage'],
                'num_users_evaluated': metrics['content_based']['num_users']
            },
            'hybrid': {
                'precision_at_5': metrics['hybrid']['precision_at_k'],
                'coverage': metrics['hybrid']['coverage'],
                'num_users_evaluated': metrics['hybrid']['num_users']
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# INFO ENDPOINT
# ============================================================================

@app.route('/api/info', methods=['GET'])
def get_info():
    """
    Get system information
    
    Example:
        GET /api/info
    """
    try:
        return jsonify({
            'service': 'Course Recommendation System API',
            'version': '1.0.0',
            'phase': 3,
            'description': 'Production-ready recommendation engine for online courses',
            'statistics': {
                'total_users': int(df_train['user_id'].nunique()),
                'total_courses': df_train['course_name'].nunique(),
                'total_enrollments': len(df_train),
                'algorithms_available': ['hybrid', 'popularity', 'content_based']
            },
            'endpoints': {
                'health_check': 'GET /health',
                'recommend': 'GET /api/recommend?user_id=<id>&algorithm=<algo>&k=<num>',
                'user_profile': 'GET /api/user/<user_id>',
                'all_courses': 'GET /api/courses',
                'course_details': 'GET /api/courses/<course_name>',
                'batch_recommend': 'POST /api/batch_recommend',
                'metrics': 'GET /api/metrics',
                'info': 'GET /api/info'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Course Recommendation System - Flask API")
    print("="*80)
    print("\n🚀 Starting API server...")
    print("📍 Server running at: http://localhost:5000")
    print("\n📚 API Documentation:")
    print("   GET  http://localhost:5000/api/info          - API info & endpoints")
    print("   GET  http://localhost:5000/health             - Health check")
    print("   GET  http://localhost:5000/api/recommend      - Get recommendations")
    print("   GET  http://localhost:5000/api/user/<id>      - Get user profile")
    print("   GET  http://localhost:5000/api/courses        - Get all courses")
    print("   POST http://localhost:5000/api/batch_recommend - Batch recommendations")
    print("\n✅ Ready to serve requests!\n")
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
