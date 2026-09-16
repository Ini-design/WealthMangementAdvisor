const CHAT_ENDPOINT = "/chat";
const HEALTH_ENDPOINT = "/health";


const queryInput = document.getElementById("queryInput");
const askBtn = document.getElementById("askBtn");
const askBtnText = document.getElementById("askBtnText");

const charCount = document.getElementById("charCount");

const answerSection = document.getElementById("answerSection");
const answerContent = document.getElementById("answerContent");
const typingIndicator = document.getElementById("typingIndicator");

const newQuestionBtn = document.getElementById("newQuestionBtn");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

const aboutBtn = document.getElementById("aboutBtn");
const aboutModal = document.getElementById("aboutModal");
const closeModal = document.getElementById("closeModal");


/* CHARACTER COUNT */

function updateCharacterCount() {

    const length = queryInput.value.length;

    charCount.textContent = `${length} / 500`;

}


/* QUICK QUESTIONS */

document.querySelectorAll(".quick-chip").forEach((chip) => {

    chip.addEventListener("click", () => {

        queryInput.value = chip.textContent.trim();

        updateCharacterCount();

        queryInput.focus();

    });

});


/* ENTER TO SUBMIT */

queryInput.addEventListener("keydown", (event) => {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        askQuestion();

    }

});


queryInput.addEventListener(
    "input",
    updateCharacterCount
);


/* ASK QUESTION */

askBtn.addEventListener(
    "click",
    askQuestion
);


async function askQuestion() {

    const query = queryInput.value.trim();


    if (!query) {

        queryInput.focus();

        return;

    }


    if (query.length > 500) {

        showError(
            "Your question is too long. Please keep it below 500 characters."
        );

        return;

    }


    answerSection.classList.remove("hidden");

    answerContent.textContent = "";

    typingIndicator.classList.remove("hidden");

    askBtn.disabled = true;

    askBtnText.textContent = "Analyzing...";


    answerSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });


    try {

        await streamAnswer(query);

    } catch (error) {

        console.error(error);

        showError(
            "Unable to connect to the financial research service. Please try again."
        );

    } finally {

        askBtn.disabled = false;

        askBtnText.textContent = "Analyze";

        typingIndicator.classList.add("hidden");

    }

}


/* SSE STREAM */

async function streamAnswer(query) {

    const response = await fetch(
        CHAT_ENDPOINT,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },

            body: JSON.stringify({
                query: query
            })
        }
    );


    if (!response.ok) {

        if (response.status === 429) {

            throw new Error(
                "Too many requests. Please wait a moment."
            );

        }


        if (response.status === 503) {

            throw new Error(
                "The financial research service is currently unavailable."
            );

        }


        throw new Error(
            `Request failed with status ${response.status}`
        );

    }


    if (!response.body) {

        throw new Error(
            "Streaming is not supported by this browser."
        );

    }


    const reader = response.body.getReader();

    const decoder = new TextDecoder("utf-8");

    let buffer = "";

    let finished = false;


    while (!finished) {

        const {
            value,
            done
        } = await reader.read();


        if (done) {

            break;

        }


        buffer += decoder.decode(
            value,
            {
                stream: true
            }
        );


        const lines = buffer.split("\n");

        buffer = lines.pop() || "";


        for (const rawLine of lines) {

            const line = rawLine.trim();


            if (!line) {
                continue;
            }


            /*
             * Backend heartbeat.
             * Example:
             * : ping
             */

            if (line.startsWith(":")) {
                continue;
            }


            /*
             * SSE data event.
             */

            if (!line.startsWith("data:")) {
                continue;
            }


            const data = line
                .slice(5)
                .trim();


            if (data === "[DONE]") {

                finished = true;

                break;

            }


            /*
             * Backend can send structured error events.
             */

            if (data.startsWith("[ERROR]")) {

                throw new Error(
                    data.replace("[ERROR]", "").trim()
                );

            }


            answerContent.textContent += data + " ";


            answerContent.scrollTop =
                answerContent.scrollHeight;

        }

    }


    /*
     * Prevent an empty answer from looking successful.
     */

    if (!answerContent.textContent.trim()) {

        throw new Error(
            "The research service returned an empty response."
        );

    }

}


/* ERROR */

function showError(message) {

    typingIndicator.classList.add("hidden");

    answerContent.textContent =
        `Sorry, ${message}`;

}


/* HEALTH CHECK */

async function checkHealth() {

    try {

        const response = await fetch(
            HEALTH_ENDPOINT,
            {
                method: "GET",
                cache: "no-store"
            }
        );


        if (!response.ok) {
            throw new Error("Health check failed");
        }


        statusDot.classList.add("online");

        statusDot.classList.remove("offline");

        statusText.textContent =
            "System online";


    } catch (error) {

        console.error(error);

        statusDot.classList.remove("online");

        statusDot.classList.add("offline");

        statusText.textContent =
            "System unavailable";

    }

}


/* NEW QUESTION */

newQuestionBtn.addEventListener(
    "click",
    () => {

        answerSection.classList.add("hidden");

        answerContent.textContent = "";

        queryInput.value = "";

        updateCharacterCount();

        queryInput.focus();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }
);


/* ABOUT MODAL */

aboutBtn.addEventListener(
    "click",
    () => {

        aboutModal.classList.remove("hidden");

    }
);


closeModal.addEventListener(
    "click",
    () => {

        aboutModal.classList.add("hidden");

    }
);


aboutModal
    .querySelector(".modal-overlay")
    .addEventListener(
        "click",
        () => {

            aboutModal.classList.add("hidden");

        }
    );


document.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Escape" &&
            !aboutModal.classList.contains("hidden")
        ) {

            aboutModal.classList.add("hidden");

        }

    }
);


/* INITIALIZE */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        updateCharacterCount();

        checkHealth();

        queryInput.focus();

    }
);
