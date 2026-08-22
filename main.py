import os
import pandas as pd
import numpy as np

class IntelligenteBookEvaluator:
    """
    Intelligent eBook Evaluation System for BTIS3043 Non-Written Final Assessment.
    Combines Predicate Reasoning (crisp query filters) with Fuzzy Reasoning 
    (suitability ranking based on multi-criteria membership functions).
    """

    def __init__(self, file_a, file_b, file_c):
        self.file_a = file_a
        self.file_b = file_b
        self.file_c = file_c

        # Fixed Scenario 1 Keyword Taxonomies
        self.s1_taxonomy = {
            'Direct AI': [
                'artificial intelligence', 'intelligent systems', 'machine learning', 
                'computer vision', 'robotics', 'expert systems', 'knowledge representation'
            ],
            'Programming Support': [
                'python', 'java', 'c++', 'c language', 'algorithm', 'data structure', 
                'software engineering', 'programming', 'code', 'coding'
            ],
            'Mathematical Support': [
                'statistics', 'probability', 'linear algebra', 'discrete mathematics', 
                'calculus', 'optimization', 'decision analysis', 'mathematics', 'math'
            ]
        }

        # Fixed Scenario 2 Keyword Taxonomy
        self.s2_taxonomy = {
            'Cybersecurity': [
                'cybersecurity', 'computer security', 'network security', 'cryptography', 
                'privacy', 'digital forensics', 'information assurance', 'secure systems', 'security'
            ]
        }

        self.df_a = None
        self.df_b = None
        self.df_c = None

    def load_and_preprocess(self):
        """
        Loads datasets A, B, and C, and standardizes text searching fields, 
        publication years, and price fields.
        """
        # Load Dataset A (Existing eBook Collection)
        self.df_a = pd.read_excel(self.file_a)
        self.df_a['search_text'] = self.df_a['Title'].fillna('').astype(str).str.lower()
        self.df_a['parsed_year'] = pd.to_numeric(self.df_a['Copyright Year'], errors='coerce')
        self.df_a['parsed_price'] = pd.to_numeric(self.df_a['Unit Net Price'], errors='coerce')

        # Load Dataset B (Academic eBook Catalogue)
        self.df_b = pd.read_excel(self.file_b)
        discipline_cols_b = [c for c in self.df_b.columns if 'Discipline' in c]
        search_cols_b = ['Title', 'Author'] + discipline_cols_b
        self.df_b['search_text'] = self.df_b[search_cols_b].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        
        # Extract year from Copyright or Pub Date
        year_b = pd.to_numeric(self.df_b['Copyright'], errors='coerce')
        pub_year_b = pd.to_datetime(self.df_b['Pub Date'], errors='coerce').dt.year
        self.df_b['parsed_year'] = year_b.fillna(pub_year_b)
        self.df_b['parsed_price'] = np.nan  # Dataset B does not contain unit price information

        # Load Dataset C (eBook Acquisition Catalogue)
        self.df_c = pd.read_excel(self.file_c)
        search_cols_c = ['Title', 'Author', 'Category', 'Discipline']
        self.df_c['search_text'] = self.df_c[search_cols_c].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
        self.df_c['parsed_year'] = pd.to_numeric(self.df_c['Copyright Year'], errors='coerce')
        self.df_c['parsed_price'] = pd.to_numeric(self.df_c['April List Price (USD)'], errors='coerce')

    # ------------------------------------------------------------------
    # Fuzzy Membership Functions (Fuzzy Sets)
    # ------------------------------------------------------------------
    @staticmethod
    def membership_recency(year, current_year=2026):
        """
        Fuzzy Membership Function for Publication Recency [0.0, 1.0]:
        - Year >= 2024: 1.0 (Highly Recent)
        - 2020 <= Year < 2024: Linear interpolation [0.7, 1.0]
        - 2015 <= Year < 2020: Linear interpolation [0.3, 0.7]
        - Year < 2015: 0.1 (Older / Less Recent)
        """
        if pd.isna(year):
            return 0.5  # Neutral default for missing values
        
        y = float(year)
        if y >= 2024:
            return 1.0
        elif y >= 2020:
            return 0.7 + 0.3 * (y - 2020) / 4.0
        elif y >= 2015:
            return 0.3 + 0.4 * (y - 2015) / 5.0
        else:
            return 0.1

    @staticmethod
    def membership_affordability(price, min_p=50.0, max_p=350.0):
        """
        Fuzzy Membership Function for Price Affordability [0.0, 1.0]:
        - Price <= $50: 1.0 (Highly Affordable)
        - Price >= $350: 0.1 (Expensive)
        - $50 < Price < $350: Linear decay function
        """
        if pd.isna(price):
            return 0.5  # Neutral default when price field is unavailable
        
        p = float(price)
        if p <= min_p:
            return 1.0
        elif p >= max_p:
            return 0.1
        else:
            return 1.0 - 0.9 * ((p - min_p) / (max_p - min_p))

    def evaluate_relevance(self, search_text, taxonomy_dict):
        """
        Evaluates Topic Relevance (mu_Relevance) and identifies domain relationships.
        
        Returns:
            tuple: (relevance_membership_score, relationship_label_string)
        """
        text = str(search_text).lower()
        matched_categories = []
        total_term_matches = 0

        for cat, keywords in taxonomy_dict.items():
            cat_matches = sum(1 for kw in keywords if kw in text)
            if cat_matches > 0:
                matched_categories.append(cat)
                total_term_matches += cat_matches

        if not matched_categories:
            return 0.0, "Uncategorized"

        # Format domain classification label
        relationship = " & ".join(matched_categories) if len(matched_categories) > 1 else matched_categories[0]

        # Fuzzy relevance membership score calculation
        relevance_score = min(1.0, 0.5 + 0.15 * total_term_matches)
        return relevance_score, relationship

    # ------------------------------------------------------------------
    # Query Engine & Result Processing
    # ------------------------------------------------------------------
    def run_scenario(self, df, dataset_label, scenario_num, taxonomy, top_n=5, w_rel=0.5, w_rec=0.3, w_aff=0.2):
        """
        Executes Predicate Matching followed by Fuzzy Reasoning for a given dataset and scenario.
        """
        all_keywords = [kw for kws in taxonomy.values() for kw in kws]
        
        # 1. Predicate Filtering (Crisp Logical Condition)
        predicate_mask = df['search_text'].apply(lambda t: any(kw in t for kw in all_keywords))
        pred_df = df[predicate_mask].copy()

        if pred_df.empty:
            print(f"\n========================================================")
            print(f" {dataset_label} | Scenario {scenario_num} - NO MATCHING RECORDS FOUND ")
            print(f"========================================================")
            return pred_df, pred_df

        # 2. Fuzzy Reasoning & Aggregation
        rel_scores = []
        relationships = []
        rec_scores = []
        aff_scores = []
        fuzzy_scores = []

        for _, row in pred_df.iterrows():
            rel_mu, rel_label = self.evaluate_relevance(row['search_text'], taxonomy)
            rec_mu = self.membership_recency(row['parsed_year'])
            aff_mu = self.membership_affordability(row['parsed_price'])
            
            # Weighted aggregation formula for fuzzy score calculation
            final_fuzzy = (w_rel * rel_mu) + (w_rec * rec_mu) + (w_aff * aff_mu)

            rel_scores.append(round(rel_mu, 3))
            relationships.append(rel_label)
            rec_scores.append(round(rec_mu, 3))
            aff_scores.append(round(aff_mu, 3))
            fuzzy_scores.append(round(final_fuzzy, 3))

        pred_df['Relationship'] = relationships
        pred_df['μ_Relevance'] = rel_scores
        pred_df['μ_Recency'] = rec_scores
        pred_df['μ_Affordability'] = aff_scores
        pred_df['Fuzzy_Score'] = fuzzy_scores

        # Sort results for Fuzzy-Enhanced ranking
        fuzzy_df = pred_df.sort_values(by=['Fuzzy_Score', 'parsed_year'], ascending=[False, False]).copy()

        # 3. Output Results Formatting
        print(f"\n========================================================")
        print(f" {dataset_label} | Scenario {scenario_num} Results ")
        print(f"========================================================")
        print(f"Total Predicate Match Count: {len(pred_df)}")

        title_col = 'Title' if 'Title' in pred_df.columns else pred_df.columns[0]
        
        # Display Predicate-Only Results (Initial order before fuzzy evaluation)
        print(f"\n-- [Predicate-Only Output (Displaying First {min(top_n, len(pred_df))})] --")
        pred_display_cols = [title_col, 'Relationship', 'parsed_year']
        if 'parsed_price' in pred_df.columns and pred_df['parsed_price'].notna().any():
            pred_display_cols.append('parsed_price')
        print(pred_df[pred_display_cols].head(top_n).to_string(index=False))

        # Display Fuzzy-Enhanced Results (Sorted by Fuzzy Score)
        print(f"\n-- [Fuzzy-Enhanced Output (Top {min(top_n, len(fuzzy_df))})] --")
        fuzzy_display_cols = [title_col, 'Relationship', 'μ_Relevance', 'μ_Recency', 'μ_Affordability', 'Fuzzy_Score']
        print(fuzzy_df[fuzzy_display_cols].head(top_n).to_string(index=False))

        return pred_df, fuzzy_df

    def execute_all(self):
        """
        Main execution pipeline running Scenario 1 and Scenario 2 on Dataset A, B, and C.
        """
        self.load_and_preprocess()

        # Execute Fixed Scenario 1
        print("\n" + "#"*75)
        print("# FIXED SCENARIO 1: AI, Programming & Mathematical Foundations")
        print("#"*75)
        self.run_scenario(self.df_a, "Dataset A (Existing Collection)", 1, self.s1_taxonomy, top_n=5)
        self.run_scenario(self.df_b, "Dataset B (Academic Catalogue)", 1, self.s1_taxonomy, top_n=5)
        self.run_scenario(self.df_c, "Dataset C (Acquisition Catalogue)", 1, self.s1_taxonomy, top_n=5)

        # Execute Fixed Scenario 2
        print("\n" + "#"*75)
        print("# FIXED SCENARIO 2: Cybersecurity & Secure Computing")
        print("#"*75)
        self.run_scenario(self.df_a, "Dataset A (Existing Collection)", 2, self.s2_taxonomy, top_n=10)
        self.run_scenario(self.df_b, "Dataset B (Academic Catalogue)", 2, self.s2_taxonomy, top_n=10)
        self.run_scenario(self.df_c, "Dataset C (Acquisition Catalogue)", 2, self.s2_taxonomy, top_n=10)


# Main Entry Point
if __name__ == "__main__":
    # Ensure dataset files exist in the working directory
    file_a = "BTIS3043_Dataset_A_Existing_eBook_Collection.xlsx"
    file_b = "BTIS3043_Dataset_B_Academic_eBook_Catalogue.xlsx"
    file_c = "BTIS3043_Dataset_C_eBook_Acquisition_Catalogue.xlsx"

    # Initialize evaluator and run complete execution
    evaluator = IntelligenteBookEvaluator(file_a, file_b, file_c)
    evaluator.execute_all()