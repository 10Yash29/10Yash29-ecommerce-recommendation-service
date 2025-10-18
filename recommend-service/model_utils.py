import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class ModelUtils:
    def __init__(self, df, user_to_idx, idx_to_user, product_to_idx, idx_to_product):
        self.df = df
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        self.product_to_idx = product_to_idx
        self.idx_to_product = idx_to_product
        

        self.df = self._filter_data(df)
        self.user_item_matrix = self._create_user_item_matrix()
        self.popular_products = self._calculate_popular_products()
    
    def _filter_data(self, df):
        """Filter DataFrame to only include users and products in the trained model"""
        try:
            filtered_df = df[
                (df['userId'].isin(self.user_to_idx.keys())) & 
                (df['productId'].isin(self.product_to_idx.keys()))
            ]
            
            logger.info(f"Filtered data: {len(filtered_df)} interactions from original {len(df)}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return df.iloc[0:0] 
    
    def _create_user_item_matrix(self):
        """Create the user-item interaction matrix"""
        try:
            if self.df.empty:
                logger.warning("No data available for matrix creation")
                return csr_matrix((len(self.user_to_idx), len(self.product_to_idx)))
            
            rows = [self.user_to_idx[user] for user in self.df['userId']]
            cols = [self.product_to_idx[product] for product in self.df['productId']]
            data = self.df['strength'].values
            
            matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(len(self.user_to_idx), len(self.product_to_idx))
            )
            return matrix
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {e}")
            return csr_matrix((len(self.user_to_idx), len(self.product_to_idx)))
    
    @lru_cache(maxsize=1)
    def _calculate_popular_products(self, n=10):
        """Calculate most popular products based on interaction frequency and strength"""
        try:
            if self.df.empty:
                logger.warning("No data available for popularity calculation")
                return list(self.product_to_idx.keys())[:n]
            
            product_popularity = (
                self.df.groupby('productId')
                .agg({
                    'strength': ['sum', 'count', 'mean']
                })
                .round(2)
            )
            
            product_popularity.columns = ['total_strength', 'interaction_count', 'avg_strength']
            
            if len(product_popularity) > 0:
                product_popularity['popularity_score'] = (
                    0.4 * product_popularity['total_strength'] / product_popularity['total_strength'].max() +
                    0.4 * product_popularity['interaction_count'] / product_popularity['interaction_count'].max() +
                    0.2 * product_popularity['avg_strength'] / product_popularity['avg_strength'].max()
                )
                
                popular_products = (
                    product_popularity
                    .sort_values('popularity_score', ascending=False)
                    .head(n)
                    .index.tolist()
                )
            else:
                popular_products = list(self.product_to_idx.keys())[:n]
            
            return popular_products
            
        except Exception as e:
            logger.error(f"Error calculating popular products: {e}")
            return list(self.product_to_idx.keys())[:min(n, len(self.product_to_idx))]
    
    def get_user_interactions(self, user_idx):
        """Get user interactions as sparse matrix"""
        user_id = self.idx_to_user[user_idx]
        user_data = self.df[self.df['userId'] == user_id]
        
        if user_data.empty:
            return csr_matrix((1, len(self.product_to_idx)))
            
        cols = [self.product_to_idx[pid] for pid in user_data['productId']]
        data = user_data['strength'].values
        rows = [0] * len(cols)
        
        return csr_matrix((data, (rows, cols)), shape=(1, len(self.product_to_idx)))
    
    def get_cold_start_recommendations(self, n=5):
        """Get popular products for new users"""
        popular_products = self.df.groupby('productId')['strength'].sum().nlargest(n)
        return popular_products.index.tolist()
