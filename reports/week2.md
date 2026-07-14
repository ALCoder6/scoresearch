# week 2 (7.4.26 - 7.11.26)

values of spectrogram_db range from -30 to 40
![histogram of db](2histogramofdb.png)

basic value comparision using .allclose() yields no matches <br>
cross correlation with threshold 0.95 yields hundreds of results <br>
while 0.99999 yields one, the match is incorrect
![spectrogram of input](2inputfigure.png)
![spectrogram of singular output](2foundfigure.png)

decided to get rid of db comparison in favor of frequency <br>
although db is better for visual comparision, values differ when calculated and range of -30 to 40 makes differences in numbers diffucult to work with <br>
frequency numbers range from 0 - 200ish and the unintellegible frequencies can be ignored
![amplitude cutoff at 1](2amplitudecutoffat1.png)

best similarity metric
define similarity of frequency 

euclidean distance formula (coordinates) & cosine similarity (dot product with normalization from 0-1)

write new correlation function using cosine similarity: dot product / product of magnitudes

    |------ all one frequency over time
    |
    |
    |
array of arrays ^^^
outer array is for each frequency, each element is array of cosine similarities for each time stamp

![algorithm on simple array](2primativecorrelation.png)

attempted on 5 seconds of match 0:00.05 - 0:05.05 and 10 seconds of full 0:00.00 - 0:10.00
![algorithm on real audio](2fullcosinesimilarity.png)

found matches on row 5 of the spreadsheet, suggests the code is working


