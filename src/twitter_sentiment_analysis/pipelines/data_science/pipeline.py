from kedro.pipeline import node, Pipeline
from twitter_sentiment_analysis.pipelines.data_science.nodes import split_data, vectorize_text, train_model, evaluate_model

def create_pipeline(**kwargs):
    
    return Pipeline(
        [
            node(
                split_data,
                inputs=['processed_data', 'parameters'],
                outputs=['X_train', 'X_test', 'y_train', 'y_test']
            ),
            node(
                vectorize_text,
                inputs=['X_train', 'X_test', 'parameters'],
                outputs=['X_train_vec', 'X_test_vec']
            ),
            node(
                train_model,
                inputs=['X_train_vec', 'y_train', 'parameters'],
                outputs='model'
            ),
            node(
                evaluate_model,
                inputs=['X_test_vec', 'y_test', 'model'],
                outputs=None
            )
        ]
    )