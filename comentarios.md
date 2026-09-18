## O que há dentro do keypoints.json

1. metadata traz as informações do vídeo e da extração: nome do arquivo de origem, fps, número de quadros processados e a taxa de detecção de mãos (hand_detection_rate, 71,5%) e de pose (pose_detection_rate, 100%).
2. frames é uma lista com um item por quadro do vídeo, cada um contendo hands (até duas mãos, cada uma com os 21 landmarks normalizados de 0 a 1) e pose (os 33 landmarks do corpo).
3. Cada landmark é um ponto {"landmark": i, "x", "y", "z", "score"}, onde score é a confiança de detecção da mão ou a visibilidade do ponto no caso da pose.