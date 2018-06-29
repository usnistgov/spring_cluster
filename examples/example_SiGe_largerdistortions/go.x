
for t in 1 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 450
do
#for c in 0.001 0.0005 0.0 -0.0005 -.001
for c in  -0.0001
#for c in 
do

echo $t " " $c

sed "s/TTT/$t/g" < job.run | sed "s/CCC/$c/g" > job.t
qsub -cwd job.t
sleep 1

done
done
