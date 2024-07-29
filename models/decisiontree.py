from .baseline import Classifier, focus_cv_mapping, Regressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import optuna
import matplotlib.pyplot as plt

class DecisiontreeClassifier(Classifier):

    def __init__(self, params):
        super().__init__(params)
        self.model = DecisionTreeClassifier()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'max_depth' : trial.suggest_int('max_depth', 3, 10),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5)
            }

            model = DecisionTreeClassifier(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=self.cv_folds).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = DecisionTreeClassifier(**study.best_params)

        if fit:
            self.fit(X, y)

        return
    
    def interpret(self, X_train, X_test, y_test, features, model_name):
        scores = self.model.feature_importances_
        feature_scores = [[score, feature] for score, feature in zip(scores, features)]
        feature_scores.sort(reverse = True)
        top_features = [feature[1] for feature in feature_scores[:10]]
        scores = [feature[0] for feature in feature_scores[:10]]
        scores = [score / sum(scores) for score in scores]
        return top_features, scores
    

class DecisiontreeRegressor(Regressor):

    def __init__(self, params):
        super().__init__(params)
        self.model = DecisionTreeRegressor()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'max_depth' : trial.suggest_int('max_depth', 3, 10),
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5)
            }

            model = DecisionTreeRegressor(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=5).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = DecisionTreeRegressor(**study.best_params)

        if fit:
            self.fit(X, y)

        return
    
    def interpret(self, X_train, X_test, y_test, features, model_name):
        scores = self.model.feature_importances_
        feature_scores = [[score, feature] for score, feature in zip(scores, features)]
        feature_scores.sort(reverse = True)
        top_features = [feature[1] for feature in feature_scores[:10]]
        scores = [feature[0] for feature in feature_scores[:10]]
        scores = [score / sum(scores) for score in scores]
        return top_features, scores