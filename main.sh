
i=0

while [ $i -le 1000000 ]
do
  echo Iteration: $i
  python ~/ml4co_dual_task/es/src/main.py item_placement
  python ~/ml4co_dual_task/es/src/main.py load_balancing
  python ~/ml4co_dual_task/es/src/main.py anonymous
  ((i++))
done
