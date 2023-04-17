# cluster_students_public
Public code to cluster students by test results. Includes fake test results as demonstration

Main analysis is run with the script cluster.py
This includes KMeans clustering and PCA. It outputs two spreadsheets "*_average_student.csv" and "*_CLUSTERS.csv".
They contain the average results of each cluster and the list of students that belong to each cluster respectively.
I also output the feature space from the PCA, although I suspect that the axis labels themselves are the more
relevant part.

AssessmentClass.py handles the test results as input
This object class is responsible for taking in the spreadsheet of test results. It cleans the data and normalizes
it so that it is ready to be passed to cluster.py

I have also included my py_tests as a reference to the tests that I ran on the Assessment Class. This will not work
as I did not include the test data used in these automated tests.
