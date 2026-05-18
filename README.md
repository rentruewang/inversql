# InverSQL


```

    +--------------+                    +-----------------+
    |  Controller  | <----------------- | Display to User |
    +------+-------+                    +--------+--------+
           |                                     |
    Process Input (CSV)           Returns Processed Data/SQL
           |                                     |
           V                                     |
    +--------------------------------------------+---------+
    |                        MODEL                         |
    |                                                      |
    |               [ Feature Extraction ]                 |
    |               (Joins based on stats)                 |
    |                         |                            |
    |                         |                            |
    |                         V                            |
    |               [ Run Decision Tree ]                  |
    |                         |                            |
    |                         V                            |
    |             [ Decompose Decision Tree ]              |
    |                         |                            |
    |                         V                            |
    |             [ Simplify / Optimize SQL ]              |
    |                                                      |
    +------------------------------------------------------+
```
