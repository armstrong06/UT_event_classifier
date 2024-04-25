
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate

def do_forward_selection_cv(X,
                            y,
                            feature_ids_to_select,
                            cv_outer,
                            inner_grid_search,
                            scoring_method,
                            larger_score_is_better=True,
                            required_feature_ids=None,
                            early_stopping_tol=-1,
                            verbose=False,
                            n_jobs=1):
    
    assert not np.any(np.isin(required_feature_ids, 
                            feature_ids_to_select)), ValueError("Required feature cannot be in the features to select")

    n_features_to_select_from = len(feature_ids_to_select)
    feat_selection = np.arange(n_features_to_select_from) #np.copy(feature_inds_to_select)

    cnt = 0
    best_mean_score = 0        
    starting_feature_size = 0
    # mean, min, max
    selected_test_scores_stats = []
    all_test_scores = np.full((n_features_to_select_from, n_features_to_select_from, 3), np.nan)
    selected_features_ids = [] 

    if required_feature_ids is not None:
        starting_feature_size = len(required_feature_ids)
        selected_features_ids = list(required_feature_ids)
        cnt += starting_feature_size
        X_req = np.copy(X[:, required_feature_ids])
        cv_scores = cross_validate(inner_grid_search, 
                                    X_req, 
                                    y, 
                                    scoring=scoring_method, 
                                    cv=cv_outer, 
                                    return_estimator=False,
                                    n_jobs=n_jobs)
        mean_cv_score = np.mean(cv_scores['test_score'])
        min_cv_score = np.min(cv_scores['test_score'])
        max_cv_score = np.max(cv_scores['test_score'])
        selected_test_scores_stats.append([mean_cv_score, min_cv_score, max_cv_score])
        best_mean_score = mean_cv_score
        print(selected_features_ids, mean_cv_score)
    else:
        selected_test_scores_stats.append([0, 0, 0])

    for it in range(n_features_to_select_from):
        for feature_ind in feat_selection:
            feature_id = feature_ids_to_select[feature_ind]
            feature_sub = feature_id
            if cnt == 0:
                X_sub = np.copy(X[:, feature_id:feature_id+1])
            else:
                feature_sub = np.concatenate([selected_features_ids, [feature_id]])
                X_sub = np.copy(X[:, feature_sub])

            cv_scores = cross_validate(inner_grid_search, 
                                        X_sub, 
                                        y, 
                                        scoring=scoring_method, 
                                        cv=cv_outer, 
                                        return_estimator=False,
                                        n_jobs=n_jobs)
            mean_cv_score = np.mean(cv_scores['test_score'])
            min_cv_score = np.min(cv_scores['test_score'])
            max_cv_score = np.max(cv_scores['test_score'])                
            all_test_scores[it, feature_ind, :] = [mean_cv_score, min_cv_score, max_cv_score]
            if verbose:
                print(feature_sub, mean_cv_score)


        if larger_score_is_better:
            feat_to_add_ind = np.nanargmax(all_test_scores[it, :, 0])
            best_it_score = np.nanmax(all_test_scores[it, :, 0])
        else:
            feat_to_add_ind = np.nanargmin(all_test_scores[it, :, 0])
            best_it_score = np.nanmin(all_test_scores[it, :, 0])

        selected_features_ids.append(feature_ids_to_select[feat_to_add_ind])
        selected_test_scores_stats.append(all_test_scores[it, feat_to_add_ind, :])
        
        feat_selection = np.delete(feat_selection, 
                                    np.where(feat_selection==feat_to_add_ind)[0][0])
        
        cnt += 1

        if not score_comparison_func(best_mean_score, 
                                    best_it_score, 
                                    larger_score_is_better,
                                    tol=early_stopping_tol) and it < n_features_to_select_from-1:
            print('Stopping early')
            all_test_scores = all_test_scores[:it+1, :]
            break

        if score_comparison_func(best_mean_score, 
                                    best_it_score, 
                                    larger_score_is_better):
            best_mean_score = best_it_score
    
    selected_test_scores_stats = np.concatenate(selected_test_scores_stats).reshape(it+2, 3)

    return selected_features_ids, selected_test_scores_stats, all_test_scores

def score_comparison_func(s_old, s_new, larger_score_is_better, tol=0):
    """Check if a new score is better than an old score.

    Args:
        s_old (_type_): _description_
        s_new (_type_): _description_
        larger_score_is_better (_type_): _description_

    Returns:
        _type_: _description_
    """
    if larger_score_is_better:
        if s_new-s_old > tol:
            return True
    else:
        if s_new-s_old < tol:
            return True
        
    return False
