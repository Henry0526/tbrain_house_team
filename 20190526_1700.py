# =======================================================================
# no_out 代表去除 total_price outlier 的資料(q1-1.5*iqr ~ q3+1.5*iqr)
# 1700分
# OLS 最小平方法
# =======================================================================
import statsmodels.api as sm

X = no_out.drop("total_price", axis=1).drop("building_id", axis=1).values
y = no_out["total_price"].values

results = sm.OLS(y,X).fit()

predict_y = results.predict(test.drop("building_id", axis=1).values)