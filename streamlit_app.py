"""
Course Recommendation System - Streamlit Web App
Phase 3: Deployment

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PICKLE LOADER
# ============================================================================

class StreamlitUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            module = __name__
        return super().find_class(module, name)

# ============================================================================
# LOAD MODELS (cached for performance)
# ============================================================================

@st.cache_resource
def load_models():
    """Load saved recommender models"""
    try:
        with open('recommender_models.pkl', 'rb') as f:
            models = StreamlitUnpickler(f).load()
        return models
    except FileNotFoundError:
        st.error("⚠️ Error: recommender_models.pkl not found!")
        st.info("Please run the Complete_Course_Recommendation_System.ipynb notebook first to generate the models.")
        st.stop()

@st.cache_data
def load_training_data():
    """Load training data for user history"""
    try:
        df = pd.read_excel('online_course_recommendation_v2.xlsx')
        df['completed'] = (df['time_spent_hours'] >= df['course_duration_hours'] * 0.5).astype(int)
        df['high_satisfaction'] = (df['rating'] >= 4).astype(int)
        return df
    except FileNotFoundError:
        st.error("⚠️ Error: online_course_recommendation_v2.xlsx not found!")
        st.stop()

# ============================================================================
# LOAD DATA
# ============================================================================

models = load_models()
df_train = load_training_data()

pop_recommender = models['popularity']
content_recommender = models['content_based']
hybrid_recommender = models['hybrid']
course_summary = models['course_summary']
metrics = models['metrics']

# ============================================================================
# HEADER & NAVIGATION
# ============================================================================

st.title("🎓 Course Recommendation System")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Get Recommendations",
    "📊 System Overview",
    "📈 Performance Metrics",
    "📚 Course Catalog",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: GET RECOMMENDATIONS
# ============================================================================

with tab1:
    st.header("Get Your Personalized Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get all user IDs
        all_users = sorted(df_train['user_id'].unique())
        
        # User selection
        st.subheader("1. Select a User")
        user_option = st.selectbox(
            "Choose how to select a user:",
            ["Select from list", "Enter user ID manually"],
            key="user_option"
        )
        
        if user_option == "Select from list":
            user_id = st.selectbox(
                "User ID:",
                all_users,
                index=0
            )
        else:
            user_id = st.number_input(
                "Enter User ID:",
                min_value=1,
                max_value=int(df_train['user_id'].max()),
                value=int(all_users[0])
            )
        
        # Validate user exists
        if user_id not in all_users:
            st.warning(f"⚠️ User {user_id} not found in training data")
            user_id = all_users[0]
    
    with col2:
        # Recommender selection
        st.subheader("2. Algorithm")
        recommender_type = st.selectbox(
            "Choose recommender:",
            ["Hybrid (Recommended)", "Popularity-Based", "Content-Based"],
            help="Hybrid: Blends popularity + content-based for best results"
        )
        
        # Map to internal names
        recommender_map = {
            "Hybrid (Recommended)": "hybrid",
            "Popularity-Based": "popularity",
            "Content-Based": "content_based"
        }
        rec_type = recommender_map[recommender_type]
        
        # Number of recommendations
        num_recs = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=10,
            value=5
        )
    
    st.markdown("---")
    
    # Get user information
    st.subheader("3. User Profile")
    
    user_history = df_train[df_train['user_id'] == user_id]
    user_courses_taken = set(user_history['course_name'].unique())
    user_avg_rating = user_history['rating'].mean()
    user_completion = user_history['completed'].mean()
    user_satisfaction = user_history['high_satisfaction'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Courses Taken", len(user_courses_taken))
    
    with col2:
        st.metric("Avg Rating Given", f"{user_avg_rating:.2f}/5.0")
    
    with col3:
        st.metric("Completion Rate", f"{user_completion*100:.1f}%")
    
    with col4:
        st.metric("Satisfaction Rate", f"{user_satisfaction*100:.1f}%")
    
    if user_courses_taken:
        st.write(f"**Courses already taken:** {', '.join(list(user_courses_taken)[:3])}")
        if len(user_courses_taken) > 3:
            st.write(f"... and {len(user_courses_taken) - 3} more")
    else:
        st.info("New user (no course history)")
    
    st.markdown("---")
    
    # Get recommendations
    st.subheader("4. Recommendations")
    
    if st.button("🚀 Get Recommendations", key="get_recs", use_container_width=True):
        
        # Select recommender
        if rec_type == "hybrid":
            recommender = hybrid_recommender
        elif rec_type == "popularity":
            recommender = pop_recommender
        else:
            recommender = content_recommender
        
        # Generate recommendations
        recommendations = recommender.recommend(user_id, user_courses_taken, k=num_recs)
        
        if recommendations:
            st.success(f"✅ Found {len(recommendations)} recommendations!")
            
            # Display recommendations
            for i, course_name in enumerate(recommendations, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {course_name}")
                        
                        # Get course info
                        course_info = course_summary.loc[course_name]
                        
                        # Display course details
                        col_a, col_b, col_c, col_d, col_e = st.columns(5)
                        
                        with col_a:
                            st.metric("⭐ Rating", f"{course_info['rating']:.2f}/5.0")
                        
                        with col_b:
                            st.metric("📚 Difficulty", course_info['difficulty_level'])
                        
                        with col_c:
                            st.metric("💰 Price", f"${course_info['course_price']:.2f}")
                        
                        with col_d:
                            st.metric("⏱️ Duration", f"{course_info['course_duration_hours']:.0f}h")
                        
                        with col_e:
                            st.metric("📜 Certification", course_info['certification_offered'])
                        
                        # Course description
                        st.write(f"**Instructor:** {course_info.get('instructor', 'Unknown')}")
        st.metric("Total Users", f"{df_train['user_id'].nunique():,}")
    
    with col2:
        st.metric("Total Courses", df_train['course_name'].nunique())
    
    with col3:
        st.metric("Total Enrollments", f"{len(df_train):,}")
    
    st.markdown("---")
    
    st.subheader("📊 How the System Works")
    
    st.markdown("""
    ### Three Recommendation Algorithms:
    
    **1. 🏆 Hybrid (Recommended)**
    - Combines popularity + content-based filtering
    - 60% weight on enrollment popularity (what most users like)
    - 40% weight on content similarity (courses similar to user's history)
    - Best overall accuracy and diversity
    
    **2. 📈 Popularity-Based**
    - Recommends most-enrolled courses
    - Simple and reliable
    - Works for cold-start (new users)
    - Less personalized
    
    **3. 🎯 Content-Based**
    - Finds courses similar to user's history
    - Based on: difficulty, price, duration, certification
    - Personalized to individual
    - May be repetitive
    
    ### Why NOT Collaborative Filtering?
    
    We tested user-user similarity and found:
    - Mean correlation between users: **-0.15** (negative!)
    - Users don't have consistent taste
    - Can't reliably predict preferences from other users
    - Hybrid approach works much better
    """)
    
    st.markdown("---")
    
    st.subheader("📚 Course Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        difficulty_dist = df_train['difficulty_level'].value_counts()
        st.bar_chart(difficulty_dist, use_container_width=True)
        st.caption("Courses by Difficulty")
    
    with col2:
        cert_dist = df_train['certification_offered'].value_counts()
        st.bar_chart(cert_dist, use_container_width=True)
        st.caption("Courses with/without Certification")

# ============================================================================
# TAB 3: PERFORMANCE METRICS
# ============================================================================

with tab3:
    st.header("Performance Metrics")
    
    st.info("""
    **Metric: Precision@5** - Out of 5 recommendations, how many did users rate 4+ stars?
    Higher = Better (max = 1.0)
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pop_precision = metrics['popularity']['precision_at_k']
        st.metric(
            "Popularity-Based",
            f"{pop_precision:.3f}",
            delta=f"{pop_precision*100:.1f}%"
        )
    
    with col2:
        content_precision = metrics['content_based']['precision_at_k']
        st.metric(
            "Content-Based",
            f"{content_precision:.3f}",
            delta=f"{content_precision*100:.1f}%"
        )
    
    with col3:
        hybrid_precision = metrics['hybrid']['precision_at_k']
        st.metric(
            "Hybrid (Winner) 🏆",
            f"{hybrid_precision:.3f}",
            delta=f"{hybrid_precision*100:.1f}%",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    st.subheader("Coverage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Popularity",
            f"{metrics['popularity']['coverage']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Content-Based",
            f"{metrics['content_based']['coverage']*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Hybrid",
            f"{metrics['hybrid']['coverage']*100:.1f}%"
        )
    
    st.caption("Coverage = % of courses recommended by at least one user")
    
    st.markdown("---")
    
    st.subheader("Evaluation Methodology")
    
    st.markdown("""
    **Test Set:** Last 30% of enrollments (holdout data)
    
    **Evaluation Process:**
    1. For each user in test set, get their training history
    2. Generate 5 recommendations using each algorithm
    3. Check how many recommendations user rated 4+ stars
    4. Calculate Precision@5 = (hits) / (recommendations)
    
    **Result:** Hybrid recommender achieved best balance of accuracy and coverage
    """)

# ============================================================================
# TAB 4: COURSE CATALOG
# ============================================================================

with tab4:
    st.header("Course Catalog")
    
    # Sorting options
    col1, col2 = st.columns(2)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["Enrollment (Popular)", "Rating (Best)", "Price (Cheapest)", "Duration (Shortest)"]
        )
    
    with col2:
        filter_difficulty = st.multiselect(
            "Filter by difficulty:",
            ["Beginner", "Intermediate", "Advanced"],
            default=["Beginner", "Intermediate", "Advanced"]
        )
    
    # Apply filters
    catalog = course_summary[course_summary['difficulty_level'].isin(filter_difficulty)].copy()
    
    # Apply sorting
    if sort_by == "Enrollment (Popular)":
        catalog = catalog.sort_values('enrollment_numbers', ascending=False)
    elif sort_by == "Rating (Best)":
        catalog = catalog.sort_values('rating', ascending=False)
    elif sort_by == "Price (Cheapest)":
        catalog = catalog.sort_values('course_price', ascending=True)
    else:  # Duration
        catalog = catalog.sort_values('course_duration_hours', ascending=True)
    
    # Display catalog
    st.markdown("---")
    
    for idx, (course_name, row) in enumerate(catalog.iterrows(), 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {idx}. {course_name}")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("⭐ Rating", f"{row['rating']:.2f}")
                
                with col_b:
                    st.metric("💰 Price", f"${row['course_price']:.2f}")
                
                with col_c:
                    st.metric("⏱️ Duration", f"{row['course_duration_hours']:.0f}h")
                
                with col_d:
                    st.metric("👥 Enrollments", f"{int(row['enrollment_numbers']):,}")
                
                # Course details
                details = f"""
                **Instructor:** {row.get('instructor', 'Unknown')}  
                **Difficulty:** {row['difficulty_level']}  
                **Certification:** {row['certification_offered']}  
                **Avg Feedback:** {row['feedback_score']:.3f}
                """
                st.markdown(details)
            
            st.markdown("---")

# ============================================================================
# TAB 5: ABOUT
# ============================================================================

with tab5:
    st.header("About This System")
    
    st.markdown("""
    ## Course Recommendation System
    
    A production-ready recommendation engine for online courses.
    
    ### Features
    
    ✅ **Three Algorithms**
    - Popularity-Based: Simple and reliable
    - Content-Based: Personalized and diverse
    - Hybrid: Best accuracy and coverage
    
    ✅ **Data-Driven**
    - Trained on 100,000 real course enrollments
    - 43,000+ users, 20 unique courses
    - Evaluated on holdout test set
    
    ✅ **Production-Ready**
    - Saved models for fast inference
    - REST API for integration
    - Web UI for manual exploration
    
    ### Technology Stack
    
    - **Backend:** Python, Pandas, Scikit-learn
    - **Frontend:** Streamlit
    - **API:** Flask
    - **Data:** Excel, Pickle
    
    ### Performance
    
    | Algorithm | Precision@5 | Coverage |
    |-----------|-------------|----------|
    | Popularity | {:.3f} | {:.1f}% |
    | Content-Based | {:.3f} | {:.1f}% |
    | **Hybrid** | **{:.3f}** | **{:.1f}%** |
    
    ### How to Use
    
    1. **Tab 1:** Get personalized recommendations for any user
    2. **Tab 2:** Learn how the system works
    3. **Tab 3:** See performance metrics
    4. **Tab 4:** Browse full course catalog
    5. **Tab 5:** This page
    
    ### Deployment
    
    **Run locally:**
    ```bash
    streamlit run streamlit_app.py
    ```
    
    **Deploy to cloud:**
    - Streamlit Cloud (free)
    - Heroku
    - AWS
    - Google Cloud
    
    ### Contact & Support
    
    For questions or feedback, see the GitHub repository.
    """.format(
        metrics['popularity']['precision_at_k'],
        metrics['popularity']['coverage'] * 100,
        metrics['content_based']['precision_at_k'],
        metrics['content_based']['coverage'] * 100,
        metrics['hybrid']['precision_at_k'],
        metrics['hybrid']['coverage'] * 100
    ))
    
    st.markdown("---")
    
    st.info("""
    **Built with ❤️ using Streamlit**
    
    This is Phase 3 of a complete recommendation system project.
    See the notebook for Phase 1 (EDA) and Phase 2 (Model Building).
    """)
