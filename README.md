# Advanced Imitation Learning
Using *simple imitation learning* we can train a car to avoid obstacles and take simple turns. At test time there is some special route that a user might want to take and for the same we cant train a model specifically. To solve this problem, we are using *conditional imitation learning* where the vehicle takes simple decisions itself but higher level commands are take by the user at run time.

* [ ] Input Method 1
    ```
    X = data_df[['center','command']].values
    y = data_df[['steer','throttle','brake']].values
    ```
* [ ] Input Method 2
    ```
    X = data_df[['center','command','speed']].values
    y = data_df[['steer','throttle','brake']].values
    ```
  
