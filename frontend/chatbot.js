Traceback (most recent call last):
  File "C:\python-pytorch\ml\train_nn.py", line 104, in <module>
    main()
    ~~~~^^
  File "C:\python-pytorch\ml\train_nn.py", line 57, in main
    X = vectorizer.fit_transform(X_text).toarray()
        ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "C:\python-pytorch\venv\Lib\site-packages\sklearn\feature_extraction\text.py", line 2104, in fit_transform
    X = super().fit_transform(raw_documents)
  File "C:\python-pytorch\venv\Lib\site-packages\sklearn\base.py", line 1329, in wrapper
    estimator._validate_params()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\python-pytorch\venv\Lib\site-packages\sklearn\base.py", line 492, in _validate_params
    validate_parameter_constraints(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        self._parameter_constraints,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.get_params(deep=False),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        caller_name=self.__class__.__name__,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\python-pytorch\venv\Lib\site-packages\sklearn\utils\_param_validation.py", line 98, in validate_parameter_constraints  
    raise InvalidParameterError(
    ...<2 lines>...
    )
sklearn.utils._param_validation.InvalidParameterError: The 'stop_words' parameter of TfidfVectorizer must be a str among {'english'}, an instance of 'list' or None. Got 'spanish' instead.