# Analysing data with SQL

This project is a part 2 of Shopify Fall 2022 Data Science Intern Challenge.
[Link to data set](https://www.w3schools.com/SQL/TRYSQL.ASP?FILENAME=TRYSQL_SELECT_ALL)

**a.How many orders were shipped by Speedy Express in total?**

```
SELECT Shippers.ShipperName, SUM(OrderDetails.Quantity) AS order_total
FROM ((Shippers
LEFT JOIN Orders ON Orders.ShipperID = Shippers.ShipperID)
LEFT JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID)
WHERE Shippers.ShipperName = 'Speedy Express'
GROUP BY Shippers.ShipperName;
```

ShipperName | order_total
--- | ---
Speedy Express | 3575 


**b.What is the last name of the employee with the most orders?**

```
SELECT TOP 1 Employees.LastName, SUM(OrderDetails.Quantity) AS amount_of_orders
FROM ((OrderDetails
LEFT JOIN Orders ON Orders.OrderID = OrderDetails.OrderID)
LEFT JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID)
GROUP BY Employees.LastName
ORDER BY 2 DESC;
```

LastName | amount_of_orders
--- | --- 
Peacock | 3232 


**c.What product was ordered the most by customers in Germany?**

```
SELECT TOP 1 Products.ProductName, SUM(OrderDetails.Quantity) AS total_orders
FROM (((Customers
LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID)
LEFT JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID)
LEFT JOIN Products ON Products.ProductID = OrderDetails.ProductID)
WHERE Customers.Country = 'Germany'
GROUP BY Products.ProductName
ORDER BY 2 DESC;
```

ProductName | total_orders
--- | --- 
Boston Crab Meat | 160 