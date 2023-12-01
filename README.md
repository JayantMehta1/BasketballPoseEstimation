# BasketballPoseEstimation


# Basketball trajectory and score/miss classification
Sample videos are in the analysis directory. To classify if a shot goes in, a y value of the frame as height is considered. If the basketball is between an xmin and xmax when aligned at this height, it is considered as a point/score. If it is not within the x range at that height, then it is classified as a miss.