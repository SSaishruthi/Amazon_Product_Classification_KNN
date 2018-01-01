# Amazon_Product_Classification
Classify products based on category using UCSD data set

Data PreProcessing

- Features selection: We need only information about product title and distinct categories the
  product belongs. So, all the other features were dropped.
- Handling null values: drop the rows with null values in Product title or categories columns.
- Data Transformation: created a new variable Category path using Categories column.
  Obtain the first path from categories list of lists, giving it top priority and add it as Category
  path. For categories like Books the title is not given. such categories should be excluded.
  Now in list of all categories, excluded the non-deepest category as more deeper categories
  will lead to more relevant search. for this, we sort our categories path and compare each
  other and retain deepest categories.
- Handling Product title: Title can be long and contain unimportant words. Normalized the
  data using Unicode retain data having different canonical meaning. Html parser to remove
  any tags. Removing stop words from product.
  
Categorization using KNN

- Dataset is split into 80 and 20 percent using random-state as 42
- Product title of entire dataset is vectorized using TF IDF Vectorizer with l2-normalization
  and ngram as (1,2)
- Using fit_transform and transform, both training and test data are vectorized
- Cosine similarity is found between train and test vector
- Trained input models are pickled along with category path for UI
- KNN is implemented. Based on cosine similarity value, top three category paths are
  retrieved and displayed on interface created using flask and HTML.
  
  RESULT
  
  KNN provided top three accurate categories 85% of the time
