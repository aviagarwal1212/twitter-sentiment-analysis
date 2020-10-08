from kedro.pipeline import node, Pipeline
from twitter_sentiment_analysis.pipelines.data_engineering.nodes import clean_raw_data, preprocess_data

def create_pipeline(**kwargs):
    
    return Pipeline(
        [
            node(
                clean_raw_data,
                inputs='tweet_data',
                outputs='cleaned_data'
            ),
            node(
                preprocess_data,
                inputs='cleaned_data',
                outputs='processed_data'
            )
        ]
    ) 