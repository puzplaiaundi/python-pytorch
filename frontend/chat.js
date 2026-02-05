// ===============================
// Configuración
// ===============================

const API_URL = "http://127.0.0.1:5000/api/ia/predict";

// ===============================
// Eventos
// ===============================
document.getElementById("enviar").addEventListener("click", enviarMensaje);
document.getElementById("mensaje").addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    enviarMensaje();
  }
});

// ===============================
// Funciones principales
// ===============================
function enviarMensaje() {
  const input = document.getElementById("mensaje");
  const texto = input.value.trim();

  if (!texto) return;

  mostrarMensaje("usuario", texto);
  input.value = "";

  fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ texto })
  })
    .then((response) => response.json())
    .then((data) => {
      procesarRespuestaIA(data);
    })
    .catch(() => {
      responderBot("⚠️ No puedo conectar con el servidor de IA.");
    });
}

function procesarRespuestaIA(data) {
  const { prediccion, confianza } = data;

  if (confianza < 0.6) {
    responderBot(
      "No estoy seguro de haber entendido tu consulta. ¿Puedes explicarla un poco más?"
    );
    return;
  }

  switch (prediccion) {
    case "saludo":
      responderBot("¡Hola! 😊 ¿En qué puedo ayudarte?");
      break;

    case "soporte":
      responderBot("Parece que tienes un problema técnico.");
      mostrarBoton("Contactar con soporte", () => {
        responderBot(
          "📞 Hemos registrado tu incidencia y el equipo de soporte se pondrá en contacto contigo."
        );
      });
      break;

    case "precio":
      responderBot("Es una consulta relacionada con precios.");
      mostrarBoton("Hablar con comercial", () => {
        responderBot(
          "💼 El departamento comercial te contactará para darte más información."
        );
      });
      break;

    case "general":
      responderBot(
        "Puedo darte información general. ¿Qué te gustaría saber exactamente?"
      );
      break;

    default:
      responderBot("No he podido clasificar tu mensaje.");
  }
}

// ===============================
// Funciones auxiliares
// ===============================
function mostrarMensaje(quien, texto) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");
  div.className = quien;
  div.textContent = texto;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function responderBot(texto) {
  mostrarMensaje("bot", texto);
}

function mostrarBoton(texto, accion) {
  const chat = document.getElementById("chat");
  const btn = document.createElement("button");
  btn.textContent = texto;
  btn.onclick = accion;
  chat.appendChild(btn);
}
