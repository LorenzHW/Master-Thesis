Question on how to use the Triplet loss function during the training of the VAE:

Based on the syntax of [this video](https://www.youtube.com/watch?v=d2XB5-tuCWU), I understand that we have to find 
"hard" triplets (A (anchor), P (positive), N (negative)). The current implementation finds positives and negatives based
on the label of the MNIST dataset. In other words: If the anchor has the digit 9 as label then positives have also the digit
9 as label. Negatives have another label (0-8 in the case of MNIST)

Based on your [previous comment](https://docs.google.com/document/d/1OE0Sa77etU9CD0gzbdqBN4ohIbYn6_reEudz-jZfJGI/edit?disco=AAAADPO2LTQ)
on July 25th, I believe we want to use different labels to find positives and negatives, namely the loss of another network NN.
Is my assumption correct? If so should we put the loss values of the classifier into different clusters? E.g.: All losses
between 1.3 and 1.7 have label x; losses between 1.7 and 2.1 have label y and so on. In order to have a classification problem
instead of a regression problem.

So my question summarized: What labels should be used for the triplet loss function?