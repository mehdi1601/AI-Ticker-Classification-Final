Ticket Classifier Documentation
Introduction
This documentation provides instructions for running the Ticket Classifier application using IntelliJ IDEA.This Applications aims at helping companies classify support tickets entered by customers, thus reducing the workload and the logistical problems that may arise when people select the wrong support ticket types from a drop-down menu.,
Prerequisites
Before using the application, make sure you have the following installed:
•	Python (3.6 or higher)
•	IntelliJ IDEA
•	Python plugin installed in IntelliJ IDEA
Getting Started
Follow these steps to get started with the Ticket Classifier application:
Importing from ZIP File
1.	Download the ZIP File:Download the ZIP file containing the application code.
2.	Extract the ZIP File:Extract the contents of the ZIP file to a folder on your computer.
3.	Open IntelliJ IDEA:Launch IntelliJ IDEA on your computer.
4.	Import the Project:
o	Open IntelliJ IDEA and select "Open" from the welcome screen.
o	Navigate to the folder where you extracted the ZIP file.
o	Select the project folder and click "Open."
Cloning from GitHub
1.	Clone the Repository:Clone the repository containing the application code to your computer. Repository URL:
bashCopy codegit clone 
2.	Open IntelliJ IDEA:Launch IntelliJ IDEA on your computer.
3.	Open the Cloned Project:
o	Open IntelliJ IDEA and select "Open" from the welcome screen.
o	Navigate to the folder where you cloned the repository.
o	Select the project folder and click "Open."
Running the Application
Follow these steps to run the Ticket Classifier application:
1.	Navigate to the Project Directory:Go to the project directory within IntelliJ IDEA.
2.	Run the Flask Application:Open the app.py file in IntelliJ IDEA and run it to start the Flask application.
3.	Access the Application:Once the Flask application is running, open a web browser and go to:
arduinoCopy codehttp://localhost:8080/
Using the Application
The Ticket Classifier application provides a simple web interface for categorizing customer support tickets. Here's how to use it:
1.	Enter Ticket Description:On the home page, type in the description of the customer support ticket.
2.	Submit Ticket Description:Click the "Classify" button to submit the ticket description for categorization.
3.	View Classification Result:The application will categorize the ticket into one of the groups, such as Billing Inquiry, Cancellation Request, Product Inquiry, Refund Request, or Technical Issue. The result will be displayed on the screen along with the entered ticket description.
4.	Classify Another Ticket:If you have more tickets to categorize, click the "Classify another ticket" link to return to the home page and enter another ticket description.
Additional Functionality
Apart from ticket classification, the application also offers the following features:
•	Word Cloud: View a visual representation of frequently used words in ticket descriptions.
•	Ticket Length Histogram: See a histogram showing the distribution of ticket lengths.
•	Learning Curve: View the learning curve of the classification model used by the application.

