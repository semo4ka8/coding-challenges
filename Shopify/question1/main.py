import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading CSV file with data and looking at it's header to understand which values do we have to work with

sheet_url = "https://docs.google.com/spreadsheets/d/16i38oonuX1y1g7C_UAmiK9GkY7cS-64DfiDMNiR41LM/edit#gid=0"
csv_url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
data = pd.read_csv(csv_url)
print(data.head())

# Looking at summary statistics for given dataset:

print(data.describe())

# To take closer look at outliers in order_amount column, let's plot the data.
# As we can see below, there are quite a few outliers in our dataset.

sns.set_style('darkgrid')
sns.boxplot(data=data, x='order_amount')
plt.title('Order amount distribution, uncleaned data')
plt.show()

# Adding a column with item price, to find any overpriced items.
# As we can see, maximum item value is 25725.00, which is way above median value of 153.00 and does not seam right,
# knowing, that all the shops are selling sneakers.

data['item_price'] = data.order_amount / data.total_items
print(data['item_price'].describe())

# Let's take a look at item prices without 25725.00 outlier.
# We can see, that there is still one item with a price outside of of our range (around 350.00).
# But as this outlier has not an absurdly high price, we can leave it as a part of our analysis,
# because it can be some very specific shoe type, like runner shoe for the long distances, that is
# using innovative technologies or very unique designer shoes.
plt.clf()
sns.set_style('darkgrid')
sns.boxplot(data=data.item_price[data.item_price < 25000])
plt.title('Item price distribution, below 25K')
plt.show()

# Let's find out, who made the highest value purchases.
# Below we can see, that all maximum orders where placed by user #607 in shop #42 with total items
# in one order 2000 pieces at 4:00 AM. That should be counted as an outlier, because such customer behaviour
# is not common and can not be counted in our statistical analysis, as is skewing our data.

print(data[data.order_amount == data.order_amount.max()])

# Let's find out, what else was user #607 ordering.
# Only purchases made by user #607 where those, with maximum total amount,
# so we can remove this user from our data, to make more accurate statistical findings.

print(data[data.user_id == 607])

# What particular shops are selling the most expensive shoes:

print('Shops selling most expensive shoes: ', data.shop_id[data.item_price > 250].unique())

# Prices in most expensive shops:

print('Prices in most expensive shops: ', data[(data.shop_id == 42) | (data.shop_id == 78)].item_price.unique())

# As shop with id #78 sells unrealistically overpriced shoes (over 25000.00), has to be removed from our data for
# analysis. Let's take a look at another expensive shop #42. Here we can see, that all other users,
# except user #607 were ordering small amount of items. So we are deciding to leave data containing orders from this
# shop overall, but to remove orders, made by user with id #607.

print(data[data.shop_id == 42])

# Removing outliers, that we found earlier:
# shop with id #78 sells unrealistically overpriced shoes (over 25000.00);
# orders made by user #607, who was ordering same extremely large amounts of high priced items from the same shop.

data_cleaned = data[(data.shop_id != 78) & (data.user_id != 607)]
print(data_cleaned.describe())

# Plotting order amount distribution
plt.clf()
sns.set_style('darkgrid')
sns.boxplot(data=data_cleaned, x='order_amount')
plt.title('Order amount distribution')
plt.show()

# Printing some summary statistics for the cleaned from outliers data:

print('Average order amount: ', np.round(np.mean(data_cleaned.order_amount), 2), 'that lays between minimum of ',
      np.min(data_cleaned.order_amount), ' and maximum of ', np.max(data_cleaned.order_amount), '.')
print('Average order contained: ', np.round(np.mean(data_cleaned.total_items)),
      ' items, with maximum ordered items per single order being ', np.max(data_cleaned.total_items), '.')
print('Average item price is: ', np.round(np.mean(data_cleaned.item_price), 2), ' in range from ',
      np.min(data_cleaned.item_price), ' to ', np.max(data_cleaned.item_price), '.')

# Plotting total orders by store distribution:

total_order_amount_by_store = data_cleaned.groupby('shop_id').order_amount.sum()

plt.clf()
sns.set_style('darkgrid')
sns.violinplot(x=total_order_amount_by_store)
plt.title('Total order amount by store')
plt.show()

print('Average total order amount for one shop is: ', np.round(total_order_amount_by_store.mean(), 2))
print('Minimum total order amount for one shop is: ', data_cleaned.groupby('shop_id').order_amount.sum().min())
print('Maximum total order amount for one shop is: ', data_cleaned.groupby('shop_id').order_amount.sum().max())

# Plotting sales count by shop(how many orders each shop made):
plt.clf()
plt.plot(data_cleaned.groupby('shop_id').count()['order_id'])
plt.title('Order count by shop')
plt.xlabel('Shop id')
plt.ylabel('Total orders count')
plt.show()

# We can see below, that shop #53 made 68 sales, which is maximum within shops in our data,
# while shop #42 made a minimum of 34 sales. Average sales count for one shop is: 50.0

order_count = data_cleaned.groupby('shop_id').count()['order_id'].reset_index().sort_values(['order_id'],
                                                                                            ascending=False).rename(
    columns={'order_id': 'sales_count'})
print(order_count)
print('Average sales count for one shop is: ', np.round(np.mean(order_count.sales_count)))

# We can see below, that shop #13 sold 136 pairs, which is maximum within shops in our data,
# while shop #42 sold a minimal 63 pairs of sneakers. Average sold items total for one shop is: 99.0 pairs of shoes.

shop_items_total = data_cleaned.groupby('shop_id').sum()['total_items'].reset_index().sort_values(['total_items'],
                                                                                                  ascending=False)
print(shop_items_total)
print('Average sold items total for one shop is: ', np.round(np.mean(shop_items_total.total_items)), 'pairs of shoes.')

# From scatter plot below, we can see, that middle priced sneakers were sold the most.
# But there is no strong relationship between price increase and sales increase.

sales_by_price = data_cleaned.groupby('item_price').total_items.sum().reset_index().sort_values(['item_price'])

plt.clf()
sns.scatterplot(x=sales_by_price.item_price, y=sales_by_price.total_items)
plt.title('Sold pairs by price')
plt.show()

# Finding customer, who ordered the most pairs of sneakers.
# Average customer ordered total of: 33.0 pairs of shoes in 30 days period,
# while customer #718 ordered total of 58 pairs, which is maximum within data.

customer_orders = data_cleaned.groupby('user_id').total_items.sum().reset_index().sort_values(['total_items'],
                                                                                              ascending=False)
print(customer_orders)
print('Average customer ordered total of: ', np.round(np.mean(customer_orders.total_items)),
      'pairs of shoes in 30 days period.')

# Finding customer, who spent the most.
# Average customer spent total of: 4979.0 on sneakers in 30 days period,
# while customer #718 spent total of 8952.0, which is maximum within data.

customer_orders_amount = data_cleaned.groupby('user_id').order_amount.sum().reset_index().sort_values(['order_amount'],
                                                                                                      ascending=False)
print(customer_orders_amount)
print('Average customer spent total of: ', np.round(np.mean(customer_orders_amount.order_amount)),
      'on sneakers in 30 days period.')

# From plot below we can see, that more customers choose to pay with credit card, compared to debit and cash payments,
# but the difference is not significant.
plt.clf()
sns.countplot(data=data_cleaned, x='payment_method')
plt.title('Payment method distribution')
plt.xlabel('Payment method')
plt.ylabel('Orders')
plt.show()

# Changing type of data in column 'created_at' to datetime, adding new column with extracted hour of purchase and
# plotting orders by hour. We can see, that in 24 hour range more orders were placed at 3 and 7 AM, as well as 5 PM (
# 17:00), but there is no significant difference in order times.

data_cleaned.loc[:, 'created_at'] = data_cleaned.created_at.astype('datetime64[ns]')
data_cleaned.loc[:, 'order_time'] = data_cleaned['created_at'].dt.hour

plt.clf()
sns.countplot(x=data_cleaned['order_time'])
plt.title('Orders by hour')
plt.xlabel('Hour')
plt.ylabel('Order count')
plt.show()
