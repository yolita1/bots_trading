/* Import d'une police moderne et lisible */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

/* Remise à zéro basique */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* Corps de page : fond sombre + dégradé */
html, body {
  height: 100%;
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(130deg, #202020, #2b2b2b);
  color: #ffffff;
}

/* Conteneur principal : on laisse le quiz prendre toute la largeur */
body {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem;
}

/* Titre principal bien visible, style « neon pink » */
h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
  font-weight: 600;
  color: #f50057; /* Rose flashy */
  text-align: center;
  text-shadow: 0 0 8px rgba(245, 0, 87, 0.7);
}

/* Sous-titres */
h2, h3 {
  font-weight: 600;
  color: #fff;
  text-shadow: 0 0 4px rgba(255, 255, 255, 0.3);
  margin-bottom: 1rem;
}

/* Paragraphe générique */
p {
  margin-top: 1rem;
  line-height: 1.5;
  font-size: 1.2rem;
}

/* =========================
   Mise en page du FORMULAIRE
   ========================= */
form {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-evenly;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 1rem 0;
  padding: 4.5rem;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.05);
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
}

.form-row {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

/* on cache le label */
.form-row label {
  display: none;
}

/* =========================
   CHAMPS DE SAISIE
   ========================= */
form input {
  min-width: 180px;
  font-size: 1.2rem;
  padding: 0.8rem 2rem;
  border: none;
  outline: none;
  border-radius: 4px;
  color: #000;
  background: #fff;
  transition: box-shadow 0.3s ease, transform 0.2s ease;
}

form input:focus {
  box-shadow: 0 0 10px #f50057;
  transform: scale(1.02);
}

/* Ordre de tabulation */
#form1 { tabindex: 1; }
#form2 { tabindex: 2; }
#form3 { tabindex: 3; }
button[type="submit"] { tabindex: 4; }



/* =========================
   BOUTON SOUMETTRE
   ========================= */
button[type="submit"] {
  cursor: pointer;
  font-size: 1.4rem;
  font-weight: 600;
  padding: 0.8rem 1.9rem;
  border: none;
  border-radius: 4px;
  background: #f50057;
  color: #ffffff;
  transition: background 0.3s, transform 0.2s;
  box-shadow: 0 0 10px rgba(245, 0, 87, 0.5);
  
}
button[type="submit"]:hover {
  background: #ff3377;
  transform: scale(1.05);
}
button[type="submit"]:active {
  transform: scale(0.97);
}

/* =========================
   MESSAGES DE FEEDBACK
   ========================= */
.correct-message {
  color: #f50057;
  font-weight: 600;
  animation: popIn 0.4s ease-out;
}
.incorrect-message {
  color: #ff4444;
  font-weight: 600;
  animation: shake 0.4s ease-in-out;
}

/* Animations */
@keyframes popIn {
  0% { transform: scale(0.8); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes shake {
  0% { transform: translateX(0); }
  20% { transform: translateX(-6px); }
  40% { transform: translateX(6px); }
  60% { transform: translateX(-6px); }
  80% { transform: translateX(6px); }
  100% { transform: translateX(0); }
}

/* =========================
   SCORE & HISTORIQUE
   ========================= */
.current-score {
  font-size: 1.4rem;
  font-weight: 600;
  color: #f50057;
  text-align: center;
  margin-top: 1rem;
  animation: glowing 2s infinite alternate;
}
@keyframes glowing {
  0% { text-shadow: 0 0 10px #f50057; }
  100% { text-shadow: 0 0 25px #f50057; }
}
ul {
  list-style: disc inside;
  margin-top: 1rem;
  line-height: 1.6;
}
