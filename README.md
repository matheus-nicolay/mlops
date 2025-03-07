# Saúde Fetal
As Cardiotocografias (CTGs) são opções simples e de baixo custo para avaliar a saúde fetal, permitindo que os profissionais de saúde atuem na prevenção da mortalidade infantil e materna. O próprio equipamento funciona enviando pulsos de ultrassom e lendo sua resposta, lançando luz sobre a frequência cardíaca fetal (FCF), movimentos fetais, contrações uterinas e muito mais.

Este conjunto de dados contém 2126 registros de características extraídas de exames de Cardiotocografias, que foram então classificados por três obstetras especialistas em 3 classes:

- Normal
- Suspeito
- Patológico

As principais variáveis de entrada incluem:  
✅ **Acelerações cardíacas**  
✅ **Movimentos fetais**  
✅ **Contrações uterinas**  
✅ **Desacelerações cardíacas severas**  

---

## **Detalhes Técnicos**  

### ** Tecnologias Utilizadas**  
📌 **Linguagem:** Python 3.9  
📌 **Frameworks:** TensorFlow, Scikit-learn, MLflow  
📌 **Serviço de Rastreamento de Experimentos:** MLflow via DagsHub  
📌 **APIs e Servidores:** FastAPI + Uvicorn  
📌 **Docker:** Contêinerização para facilitar a implantação  

---

### ** Endpoints da API**
Após rodar a API, acesse a documentação interativa do FastAPI em:  
📌 `http://localhost:80/docs`  

Exemplo de requisição via **cURL**:  
```bash
curl -X POST "http://localhost:80/predict" -H "Content-Type: application/json" -d '{
  "aceleracoes": 0.5,
  "movimentos_fetais": 2,
  "contracoes": 0.8,
  "desaceleracoes_severas": 0.1
}'