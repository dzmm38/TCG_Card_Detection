## TCG Card Detection - Project
This project ist linked to the TCM (Trading Card Collection Manager) and serves as a proof of Concept for the functionality of correctly identifing a certain Card from a video stream.

## General
At the moment this project uses opencv methods and techniques to identify an object (here a trading card) and then correctly identify the type of card. For this to work there needs to be a relative good picture of the cards artwork in the data folder unter set_1. The Programm then tries to find similar points for every card in this folder and picks the one with the most. 

## Future Work
I also tried some ML techniques and AI Methods like CNN´s but there seems to be to few test data for specific Card Games. Another Problem is the scalability of these Models because the Games are rapidly growing and releasing new cards. 

In the Future i want to futher investigate the usage of Neural Networks for classification and detection (YOLO) for this usecase because my approach works for the amount of cards tested (204) but I think it will become problematic if it´s used with tens of thousands of cards.