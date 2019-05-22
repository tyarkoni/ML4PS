import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from sklearn.model_selection import learning_curve


def get_features(data, *args, n=None):
    ''' Return specified features or feature groups for a subsample.
    
    Args:
        data (DataFrame): DataFrame to subsample.
        args (list): Positional args containing the names of
            variables/groups to return. Each element must be one of
            'domains', 'facets', 'items', or the name of an existing
            column.
        n (int): Number of cases to return. If None, keeps all rows.

    Returns: A list of pandasDataFrames and/or Series, in the same
        order as *args.
    '''

    if n is not None:
        data = data.sample(n)

    results = []
    for name in args:
        if name == 'domains':
            results.append(data.iloc[:, -5:])
        elif name == 'facets':
            results.append(data.iloc[:, -35:-5])
        elif name == 'items':
            results.append(data.iloc[:, -335:-35])
        else:
            results.append(data[name])

    return results


def plot_learning_curves(estimators, X_sets, y, train_sizes, labels=None,
                         errors=True, **kwargs):
    ''' Generate multi-panel plot displaying learning curves for multiple
    predictor sets and/or estimators.
    
    Args:
        estimators (Estimator, list): A scikit-learn Estimator or list of
            estimators. If a list is provided, it must have the same number of
            elements as X_sets.
        X_sets (NDArray-like, list): An NDArray or similar object, or list. If
            a list is passed, it must have the same number of elements as
            estimators.
        y (NDArray): a 1-D numpy array (or pandas Series) representing the
            outcome variable to predict.
        train_sizes (list): List of ints providing the sample sizes at which to
            evaluate the estimator.
        labels (list): Optional list of labels for the panels. Must have the
            same number of elements as X_sets.
        errors (bool): If True, plots error bars representing 1 StDev.
        kwargs (dict): Optional keyword arguments passed on to sklearn's
            `learning_curve` utility.
    '''
    # Set up figure
    n_col = len(X_sets)
    fig, axes = plt.subplots(1, n_col, figsize=(4.5 * n_col, 4), sharex=True,
                             sharey=True)
    
    # If there's only one subplot, matplotlib will hand us back a single Axes,
    # so wrap it in a list to facilitate indexing inside the loop
    if n_col == 1:
        axes = [axes]

    # If estimators is a single object, repeat it n_cols times in a list
    if not isinstance(estimators, (list, tuple)):
        estimators = [estimators] * n_col
    
    cv = kwargs.pop('cv', 10)

    # Plot learning curve for each predictor set
    for i in range(n_col):
        ax = axes[i]
        results = learning_curve(estimators[i], X_sets[i], y,
                                 train_sizes=train_sizes, shuffle=True,
                                 cv=cv, **kwargs)
        train_sizes_abs, train_scores, test_scores = results
        train_mean = train_scores.mean(1)
        test_mean = test_scores.mean(1)
        ax.plot(train_sizes_abs, train_mean, 'o-', label='Train',
                lw=3)
        ax.plot(train_sizes_abs, test_mean, 'o-', label='Test',
                lw=3)
        axes[i].set_xscale('log')
        axes[i].xaxis.set_major_formatter(ScalarFormatter())
        axes[i].grid(False, axis='x')
        axes[i].grid(True, axis='y')
        if labels is not None:
            ax.set_title(labels[i], fontsize=16)
        ax.set_xlabel('Num. obs.', fontsize=14)
        
        if errors:
            train_sd = train_scores.std(1)
            test_sd = test_scores.std(1)
            ax.fill_between(train_sizes, train_mean - train_sd,
                            train_mean + train_sd, alpha=0.2)
            ax.fill_between(train_sizes, test_mean - test_sd,
                            test_mean + test_sd, alpha=0.2)
    
    # Additional display options
    plt.legend(fontsize=14)
    plt.ylim(0, 1)
    axes[0].set_ylabel('$R^2$', fontsize=14)
    axes[-1].set_ylabel('$R^2$', fontsize=14)
    axes[-1].yaxis.set_label_position("right")


# Read in the Johnson et al. (2015) data
def read_IPIP_data(max_lines=None):

    # Define the non-item fields
    fields = [
        ('CASE', 6),
        ('SEX', 1),
        ('AGE', 2),
        ('SEC', 2),
        ('MIN', 2),
        ('HOUR', 2),
        ('DATE', 2),
        ('MONTH', 2),
        ('YEAR', 3),
        ('COUNTRY', 11),   
    ]

    # Parse the raw data into a list of lists
    f = Path('/', 'Users', 'tal', 'Downloads', 'IPIP300.dat.gz')
    data = []
    fc = gzip.open(f, 'rt')

    for l in fc.readlines():
        vals = []
        c = 0
        for (_, chars) in fields:
            vals.append(l[c:(c + chars)])
            c += chars
        item_scores = l[c:].strip()
        # for simplicity, skip cases with any missing value.
        # in the real world, we might want to impute.
        if '0' in item_scores:
            continue
        scores = [int(i) for i in list(item_scores)]
        vals.extend(scores)
        data.append(vals)
        
        if max_lines is not None and len(data) == max_lines:
            break
    
    # make pandas DataFrame
    cols = [t[0] for t in fields] + [f"I{i}" for i in range(1, 301)]
    df = pd.DataFrame(data, columns=cols)
    
    # Convert columns to int types
    for c in ['SEX', 'AGE', 'SEC', 'MIN', 'HOUR', 'DATE', 'MONTH', 'YEAR']:
        df[c] = df[c].astype(int)
    
    # strip whitespace from country code
    df['COUNTRY'] = df['COUNTRY'].str.strip()

    # read in the item key
    f =  Path('data', 'IPIP-NEO-ItemKey.xls')
    key = pd.read_excel(f).set_index('Full#')

    # Extract item scores and count 0s (which are NaNs) in each row.
    # Then we can divide the dot product of the score matrix and the
    # NaN matrix element-wise to obtain means based on valid responses.
    X = df.iloc[:, -300:].values.astype(float)
    valid_resps = (X != 0).astype(int)
    
    # Compute facet scores. For convenience, we ignore NaNs rather than
    # imputing; note that this will add some bias to the data in case
    # items are not MAR (but so would imputation).
    facets = key.sort_index()['Facet'].astype('category')
    mat = np.zeros((300, 30))
    mat[(range(300), facets.cat.codes)] = 1
    scale_scores = X.dot(mat) / valid_resps.dot(mat)
    scale_df = pd.DataFrame(scale_scores, columns=facets.cat.categories)

    # Concatenate DFs
    data = pd.concat([df, scale_df], axis=1)
   
    # Compute domain scores
    domains = [
        ('Neuroticism', ['Anxiety', 'Anger', 'Depression',
                         'Self-Consciousness', 'Immoderation',
                         'Vulnerability']),
        ('Extraversion', ['Friendliness', 'Gregariousness', 'Assertiveness',
                          'Activity Level', 'Excitement-Seeking',
                          'Cheerfulness']),
        ('Openness', ['Imagination', 'Artistic Interests', 'Emotionality',
                      'Adventurousness', 'Intellect', 'Liberalism']),
        ('Agreeableness', ['Trust', 'Morality', 'Altruism', 'Cooperation',
                           'Modesty', 'Sympathy']),
        ('Conscientiousness', ['Self-Efficacy', 'Orderliness', 'Dutifulness',
                               'Achievement-Striving', 'Self-Discipline'])
    ]
    for name, scales in domains:
        data[name] = data[scales].mean(1)
 
    return data