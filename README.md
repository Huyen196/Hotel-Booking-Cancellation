# HOTEL'S BOOKING CANCELLATION
![](https://images.pexels.com/photos/338504/pexels-photo-338504.jpeg?cs=srgb&dl=pexels-thorsten-technoman-109353-338504.jpg&fm=jpg)

Source: Pexels

In the past few years, both the City Hotel and Resort Hotel have experienced significant increases in their cancellation rates. As a result, both hotels are currently facing a range of challenges, such as reduced revenue and underutilized hotel rooms. Therefore, the top priority for both hotels is to reduce their cancellation rates, which will enhance their efficiency in generating revenue. 

This project aims to use the hotel dataset to identify the factors effect to guests decide cancel their booking and develop a predictive model to reduce this situation. A model figure out the factors which affect to the hotel’s booking cancellation, along with actionable insights to reduce cancelation rates.

With this data, it’s possible to support the hotels have a model to predict if a guest will actually come. This can help hotels to plan things like personel and food requirements. Moreover, they can figure out how to reduce the number of booking cancellation and increase their revenue.

## Dataset

**Source**: Public dataset from Kaggle (Hotel Revenue: https://www.kaggle.com/datasets/govindkrishnadas/hotel-revenue)

**Description:**

This dataset contains 141,314 observations for a City Hotel and a Resort Hotel. Each observation represents a hotel booking between the 1st of July 2018 and 31st of August 2020, including booking that effectively arrived and booking that were canceled.

**Structure:**

•	Hotel: One of the hotels is a resort hotel and the other is a city hotel.

•	is_canceled: Value indicating if the booking was canceled (1) or not (0).

•	lead_time: Number of days that elapsed between the entering date of the booking into the PMS and the arrival date

•	arrival_date_year: Year of arrival date.

•	arrival_date_month: Month of arrival date with 12 categories: “January” to “December”.

•	arrival_date_week_number: Week number of the arrival date.

•	arrival_date_day_of_month: Day of the month of the arrival date.

•	stays_in_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel.

•	stays_in_week_nights: Number of week nights (Monday to Friday) the guest stayed.

•	adults: Number of adults

•	children: Number of Childern

•	babies: Number of Babies

•	meal: BB – Bed & Breakfast

•	country: Country of origin.

•	market_segment: Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”

•	distribution_channel: Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”

•	is_repeated_guest: Value indicating if the booking name was from a repeated guest (1) or not (0)

•	previous_cancellations: Number of previous bookings that were cancelled by the customer prior to the current booking

•	previous_bookings_not_canceled: Number of previous bookings not cancelled by the customer prior to the current booking

•	reserved_room_type: Code of room type reserved. Code is presented instead of designation for anonymity reasons.

•	assigned_room_type: Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons

•	booking_changes: Number of changes/amendments made to the booking.

•	deposit_type: No Deposit – no deposit was made; Non Refund – a deposit was made in the value of the total stay cost; Refundable – a deposit was made with a value under the total cost of stay.

•	agent: ID of the travel agency that made the booking

•	company: ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons

•	days_in_waiting_list: Number of days the booking was in the waiting list before it was confirmed to the customer

•	customer_type: Group – when the booking is associated to a group; Transient – when the booking is not part of a group or contract, and is not associated to other transient booking; Transient-party – when the booking is transient, but is associated to at least other transient booking

•	adr: Average Daily Rate (Calculated by dividing the sum of all lodging transactions by the total number of staying nights)

•	required_car_parking_spaces: Number of car parking spaces required by the customer

•	total_of_special_requests: Number of special requests made by the customer (e.g. twin bed or high floor)

•	reservation_status: Check-Out – customer has checked in but already departed; No-Show – customer did not check-in and did inform the hotel of the reason why

•	reservation_status_date: Date at which the last status was set. This variable can be used in conjunction with the Reservation Status to understand when was the booking canceled or when did the customer checked-out of the hotel

## Data exploration

### Data Cleaning and handling value

Firstly, imported the dataset to colab. The dataset has 3 main sheets: 2018, 2019 and 2020. For easily to analyze, use "concat" to merge 3 sheets become 1 dataframe and called “df_full”.
Secondly, I checked the dataset as below:

•	Missing value: drop 2 columns are "company, agent" because they are missing a lot value and not necessary  to analyze.

•	Data type: convert all column's type

•	Check statistics of the dataset

•	Check various categories present in the different categorical columns

### Exploratory Data Analysis (EDA)

 1.	Reservation Status and Revenue / Loss

![image](https://github.com/user-attachments/assets/fd8dd104-e18c-4ed2-90bb-71a1d968d7e6)

![image](https://github.com/user-attachments/assets/db615701-a26e-4163-adec-0841aee25ed0)

The tables give the information about the total bookings in 3 years (both checked-out and canceled) and  the Revenue / Loss of both hotels from 2018 to 2020. With total around 140,000 bookings, the canceled around 51K bookings. It can be seen that the hotel has lost approximately $18.56 million in revenue due to these canceled bookings. It is evident that a substantial portion of reservations remains unaffected by cancellations. Notably, nearly 38% of guests have chosen to cancel their reservations, and this has a noteworthy impact on the hotels' revenue.

![image](https://github.com/user-attachments/assets/703393b2-5250-4750-9bd9-b3010697527e)

![image](https://github.com/user-attachments/assets/80b561e0-dba2-49d7-8a89-d85cca918733)

The dataset seperated into 2 hotels: City Hotel and Resort Hotel. The bar chart gives the information about the number of canceled booking in 3 years of each hotels. In general, the total canceled bookings tend to increase through 3 years. In comparison to Resort Hotel, City Hotel have more canceled bookings. Its possible that city hotel are more expensive that resort city.
The line graph described the total cancelation booking of each months. The number of cancelation at Hotel City booking was higher than Resort City. It is clearly to see that both hotels tend to increase rapidly from January to Agust, reached a peak at August.  After that, it was decreased until the end of year. 

Comparing with the bar chart “Total booking by month” above, it is more clearly to see months which have the highest and the lowest  reservation levels based on their status. It's evident that the month of August stands out, having the highest numbers of both confirmed and canceled reservations. January has the fewest confirmed reservations. Based on the charts, we can see that the peak time for room cancellations is likely to be between May and October.

2.	Average Daily Rate (ADR)

![image](https://github.com/user-attachments/assets/58c59234-7a42-41b4-b0ef-5a5d64334178)

![image](https://github.com/user-attachments/assets/3606cc93-1897-4685-93b9-5f216d0c65fd)

This lines charts illustrate the average price (canceled and check-out) of 2 hotels. In general, the price in City Hotel is more expensive than Resort Hotel for almost months. However, the prices in the Resort Hotel are much higher during the summer (highest price in Agust with 175$) and prices of City Hotel varies less and is most expensive during Spring and Autumn. In addition, the cancellations are most frequent when prices are at their highest and least common when prices are at their lowest. Only August at City Hotel was different with other. Although the ADR decreased, the total cancelation booking was highest.

Consequently, the price of accommodation appears to be one of the factor influencing cancellations.

3. Customer types

![image](https://github.com/user-attachments/assets/749bb2de-0fc4-42f6-9a17-0f9dfc98a039)

![image](https://github.com/user-attachments/assets/cb1520c2-7109-43ef-a054-5eebc17dfdba)

It can be seen that the Transient account for the highest number, approximately 71% (includes canceled and check-out), and this is also the customer type with the highest number of room cancellations in both hotels. Most of the cancellation booking of Transient belongs to Hotel City. While Transient-Party type ranks second in the number of room cancellations, it is only 37% compared to the number of bookings that have been check-out. 

In addition, “contract” and “group” are 2 customer types which have the lowest number of room cancellation, especially group type doesn’t have any bookings were cancelled.

4. Deposit types

![image](https://github.com/user-attachments/assets/564a20e5-f4cc-4362-91ff-ba1cff7b055f)

![image](https://github.com/user-attachments/assets/79761ec6-6cd5-4cb9-b37b-20db1ab1c1ee)

There are 2 popular deposit types: No Deposit and Non Refund. It can be seen that the “Non Refund” and the “is_canceled” column are correlated in a counter-intuitive way. Over 99 % of people who paid the entire amount upfront canceled. While the number of cancellation booking of “No Deposit” around 33K bookings (37.5%), the number of booking confirmed are 88K bookings. In addition, half of cancellation booking were canceled at City Hotel. It’s seem that “No Deposit” is chose more than Non Refund at 2 hotels. 

Based on the data, even though the non-refundable room cancellation rate over 99%, the hotel still profits from these bookings because according to the hotel's policy, customers are still required to pay a fee for cancellations.

5. Market Segment

![image](https://github.com/user-attachments/assets/e2c64acd-4c39-4783-993a-f07c8438b948)

![image](https://github.com/user-attachments/assets/e2472650-da50-4e0f-b73f-bcac1f727436)

The analysis of booking cancellations by market segment reveals important insights into the behavior of customers in different market segments. By understanding which market segments have the highest cancellation rates, hotels can tailor their strategies to minimize cancellations and optimize revenue.

The bar charts shows the sources from which guests are making hotel reservations, including Direct, Groups, Corporate, Online Travel Agencies, and Offline Travel Agents. Approximately 44% of clients make reservations through online travel agencies, around 22% for Offline Travel Agents, 19% come through group bookings, 10% of clients choose to book hotels directly by visiting them in person and making reservations, only 4% of Corporate. Most of them have the total bookings check-out more than canceled booking. However, only “Group” different. The number of canceld booking are 16K while check-out booking are 10K bookings.

From the data, most of booking cancelation of all market segment focus on City Hotel. Based on that, hotels can identify that Online/ Offline TA/TO and Group market segment have a higher propensity to cancel bookings. This information can help hotels implement targeted marketing campaigns or policies to reduce cancellations in these segments. By focusing on the market segments with the highest cancellation rates, hotels can potentially increase customer satisfaction, improve booking retention, and enhance overall profitability.

6. Lead time

According to SiteMinder’s Hotel Booking Trends, lead time is the time between a guest making their booking and arriving at your hotel for their stay. The booking lead time is the window between purchase and experience in which many things can happen, including a cancellation, extra expenditure, modifications, or requests.

![image](https://github.com/user-attachments/assets/44101ecb-637b-43d3-9437-0f1d7f5160f3)

![image](https://github.com/user-attachments/assets/9338ca46-c0c6-43fa-9c32-b9509e4c0f40)

![image](https://github.com/user-attachments/assets/d03deb6c-e84d-498a-b5d0-e8378aae3582)

![image](https://github.com/user-attachments/assets/2bb33b85-fd3a-4bc7-a4f5-b7181f124f85)

In general, bookings made a few days before the arrival date are rarely canceled, whereas bookings made over one month in advance are canceled very often. For the cancellation, the lead time booking at Resort Hotel is longer than City Hotel. 

For the City Hotel, customers tend to cancel with bookings were booked between 31 to 87 days before the arrival date, with the peak being around 1 month before. On the other hand, for Resort Hotel, the lead time for bookings is quite long, approximately close to a year. Longer lead times for Resort Hotel can lead to more uncertainties or changes in customer plans, resulting in higher cancellation rates compared to city hotels where plans might be more concrete closer to the arrival date.

7.	Correlation

![image](https://github.com/user-attachments/assets/3329c78b-82fd-4ac7-bc4e-4c7262b2adab)

Because the number of columns in the dataset is quite large (32 columns), the author has removed factors that do not have much impact on the cancellation of hotel bookings by customers.

Based on the correlation, there are some factors that have a higher impact on room cancellations compared to the remaining factors such as: lead time, deposit type, market segment, etc. With lead time, if the confirmed date is further away from the arrival date, the cancellation rate is higher (0.43). 

There are 2 types of deposits. When guests book with a non-refundable policy, the capable of room cancellation is higher. Conversely, the cancellation rate decreases when the number of guests choosing “No deposit” increases.

For market segment, although “Online TA” and “Offline TA/TO” are 2 market segment have the highest number of cancellation booking, it’s not affect too much to guest’s decision. On the other hand, the “Group” market segment has more affect to the number of cancellation booking than others. The “Direct” segment is one of the good market segment. The higher from “Direct” segment increase, the lower the number of cancellations.

Beside that, with the guests who had cancellation booking before, they tend to continue in the future. That’s thing will increase the total booking cancellation of hotels In addition, if the guests send more request to the hotels, the capable of cancel booking will lower than the guests who not or less requests

8.	Regression

![image](https://github.com/user-attachments/assets/7cb4019b-4824-4722-9791-8cf24c0964c5)

From data of OLS Regresstion, the R-squared is 0.333 which mean just explain around 33% the changes in the independent variable ( lead time, market segment, deposit type,etc) with the dependent variable (is canceled). Due to the relatively low R-squared value, this predictive model is not accurate enough.

## Conclusion and Recommendation

After analyzing the data, we can draw conclusions and suggest solutions to reduce the number of room cancellations as follows:

•	The City hotel experiences a higher ratio of cancellations compared to the Resort hotel. Therefore, hotels should consider offering competitive room price discounts on weekends and holidays.

•	Increasing prices are associated with a higher rate of cancellations. To mitigate reservation cancellations, hotels could refine their pricing strategies by offering reduced rates for specific locations and providing discounts to customers.

•	Stay competitive with your pricing while maintaining the quality of the services. Offer promotions, discounts, and loyalty programs to attract repeat customers.

•	From May to August, hotels can launch marketing campaigns with attractive offers to boost their revenue, especially since cancellations tend to peak during this period.

•	Utilize a reliable online booking platform that allows guests to make reservations easily. Provide accurate and high quality photos of hotel to reduce cancellations.

•	Encourage satisfied guests to leave positive reviews on platforms like TripAdvisor, Agoda, or Google Reviews. Respond to reviews, whether positive or negative, to show the commitment to customer satisfaction.

•	Encourage guests book a room directly with hotels. This way helps hotels can contact directly with guests and don’t need to pay commission for Travel Agent.

•	Effective Cancellation Management: Recognizing the emerging trend of post-cancellation payments, viewed as an opportunity to turn challenges into advantages from a business standpoint. By employing astute strategies aimed at reducing cancellations, including transparent terms and conditions, considerate reminders, and potentially introducing incentives for guests committed to their reservations, we can create a win-win scenario. This approach not only enhances the guest experience but also guarantees a consistent revenue flow for the business.






