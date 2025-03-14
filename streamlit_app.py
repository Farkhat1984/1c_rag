import streamlit as st
import os
import sys
from openai import OpenAI
import anthropic
import hashlib

# Добавляем корневую директорию проекта в путь поиска модулей
sys.path.append(os.path.abspath(".."))

# Импортируем компоненты из проекта
from config.config import EMBEDDING_MODEL, DEEPSEEK_API_KEY
# Добавим импорт ключа API для Anthropic
from config.config import ANTHROPIC_API_KEY
from retrieval.llm_service import call_llm_api
from data_processing.text_processor import TextProcessor
from data_processing.image_processor import ImageProcessor
from data_processing.chunking import DocumentChunker
from data_processing.summarization import TextSummarizer
from database.sqlite_manager import SQLiteManager
from database.qdrant_manager import QdrantManager
from embbeding.embbedings import TextEmbedder
from retrieval.retrieval import Retriever
from retrieval.context_builder import ContextBuilder
from utils.logger import setup_logger

logger = setup_logger("streamlit_app")

# Инициализация состояния сессии
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "components_initialized" not in st.session_state:
    st.session_state.components_initialized = False
if "error_state" not in st.session_state:
    st.session_state.error_state = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "anthropic"  # По умолчанию используем Anthropic
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# Хардкод списка пользователей
# Пароли хранятся в виде хеша
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("Ckdshfh231161!".encode()).hexdigest(),
        "role": "admin",
        "full_name": "Администратор Системы"
    },
    "user1": {
        "password_hash": hashlib.sha256("metalogicuser1!".encode()).hexdigest(),
        "role": "user1",
        "full_name": "Иван Петров"
    },
    "user2": {
        "password_hash": hashlib.sha256("metalogicuser2!".encode()).hexdigest(),
        "role": "user2",
        "full_name": "Мария Сидорова"
    },

}


def verify_password(username, password):
    """Проверяет правильность пароля для указанного пользователя"""
    if username not in USERS:
        return False

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == USERS[username]["password_hash"]


def login_user():
    """Страница входа в систему"""
    st.title("1С RAG-ассистент - Вход в систему")

    with st.form("login_form"):
        username = st.text_input("Имя пользователя")
        password = st.text_input("Пароль", type="password")
        submit_button = st.form_submit_button("Войти")

        if submit_button:
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "username": username,
                    "role": USERS[username]["role"],
                    "full_name": USERS[username]["full_name"]
                }
                st.success(f"Добро пожаловать, {USERS[username]['full_name']}!")
                st.rerun()
            else:
                st.error("Неверное имя пользователя или пароль")



# Initialize components
@st.cache_resource(show_spinner=False)
def initialize_components():
    try:
        # Получаем логгер для этой функции
        import logging
        logger = logging.getLogger("streamlit_app")

        text_processor = TextProcessor()
        image_processor = ImageProcessor()
        chunker = DocumentChunker()
        summarizer = TextSummarizer()
        sqlite_manager = SQLiteManager()
        qdrant_manager = QdrantManager()
        text_embedder = TextEmbedder(EMBEDDING_MODEL)
        retriever = Retriever(qdrant_manager, sqlite_manager, text_embedder)
        context_builder = ContextBuilder()

        logger.info("Все компоненты успешно инициализированы")

        components = {
            "text_processor": text_processor,
            "image_processor": image_processor,
            "chunker": chunker,
            "summarizer": summarizer,
            "sqlite_manager": sqlite_manager,
            "qdrant_manager": qdrant_manager,
            "text_embedder": text_embedder,
            "retriever": retriever,
            "context_builder": context_builder,
            "logger": logger  # Передаем логгер вместе с компонентами
        }

        st.session_state.components_initialized = True
        return components

    except Exception as e:
        import logging
        logger = logging.getLogger("streamlit_app")
        logger.error(f"Ошибка при инициализации компонентов: {str(e)}")
        st.session_state.error_state = True
        return None


def get_components():
    with st.spinner("Инициализация компонентов..."):
        return initialize_components()


def query_rag_system(query, components):
    # Получаем логгер из компонентов или создаем новый
    if components and "logger" in components:
        logger = components["logger"]
    else:
        import logging
        logger = logging.getLogger("streamlit_app")

    try:
        logger.info(f"Обработка запроса: {query}")

        # Поиск релевантных фрагментов и изображений
        query_result = components["retriever"].process_query(query)
        logger.info(
            f"Найдено фрагментов: {len(query_result.get('chunks', []))} и изображений: {len(query_result.get('images', []))}")

        # Формирование промпта для LLM
        prompt = components["context_builder"].build_prompt(
            query=query,
            chunks=query_result.get("chunks", []),
            images=query_result.get("images", [])
        )

        # Вызов LLM API
        answer = call_llm_api(prompt, components)

        chunks = query_result.get("chunks", [])
        images = query_result.get("images", [])

        logger.info("Запрос успешно обработан")
        return answer, chunks, images
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        st.session_state.error_state = True
        return f"Ошибка при обработке запроса: {str(e)}", [], []


def get_mime_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    elif ext == '.gif':
        return 'image/gif'
    else:
        return 'application/octet-stream'


def handle_user_input(user_input):
    # Получаем логгер
    import logging
    logger = logging.getLogger("streamlit_app")

    if not user_input:
        logger.warning("Получен пустой запрос")
        return

    logger.info(f"Получен запрос от пользователя: {user_input}")
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "user": st.session_state.current_user["username"]
    })

    # Получаем компоненты
    components = get_components()

    if components is None:
        logger.error("Не удалось инициализировать компоненты системы")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Не удалось инициализировать компоненты системы. Проверьте логи.",
            "chunks": [],
            "images": []
        })
        st.session_state.error_state = True
        return

    # Отображаем сообщение пользователя
    with st.chat_message("user"):
        st.write(user_input)

    # Обрабатываем запрос и отображаем ответ в потоковом режиме
    with st.chat_message("assistant"):
        query_result = components["retriever"].process_query(user_input)
        chunks = query_result.get("chunks", [])
        images = query_result.get("images", [])

        # Формирование промпта для LLM
        prompt = components["context_builder"].build_prompt(
            query=user_input,
            chunks=chunks,
            images=images
        )

        # Вызов LLM API в потоковом режиме
        answer = call_llm_api(prompt, components)

        # Отображаем релевантные изображения, если они есть
        if images:
            with st.expander("Релевантные изображения"):
                cols = st.columns(min(3, len(images)))
                for i, img in enumerate(images[:3]):
                    with cols[i % 3]:
                        file_path = img.get("file_path", "")
                        if os.path.exists(file_path):
                            try:
                                # Отображение изображения с адаптивной шириной
                                st.image(file_path, caption=img.get("title", ""), use_container_width=True)
                                # Кнопка для скачивания оригинала
                                with open(file_path, "rb") as file:
                                    mime_type = get_mime_type(file_path)
                                    st.download_button(
                                        label="Скачать изображение",
                                        data=file,
                                        file_name=img.get("file_name", "image"),
                                        mime=mime_type
                                    )
                            except Exception as e:
                                st.error(f"Не удалось отобразить изображение: {str(e)}")
                        else:
                            st.warning(f"Изображение не найдено: {file_path}")

    # Добавляем ответ в историю чата для будущего отображения
    logger.info(f"Добавление ответа от ассистента в историю чата")
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "images": images
    })


def chat_page():
    st.title("Чат с 1С-ассистентом")

    # Показываем информацию о текущем пользователе
    st.sidebar.subheader("Информация о пользователе")
    st.sidebar.info(f"""
    **Пользователь**: {st.session_state.current_user['full_name']}  
    **Роль**: {st.session_state.current_user['role'].capitalize()}
    """)

    # Кнопка выхода из системы
    if st.sidebar.button("Выйти из системы"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()

    if not st.session_state.components_initialized and not st.session_state.error_state:
        with st.spinner("Инициализация системы..."):
            components = get_components()
            if components is None:
                st.error("Не удалось инициализировать компоненты системы. Проверьте логи.")
                st.session_state.error_state = True

    # Отображение истории чата (все сообщения кроме последнего)
    for message in st.session_state.chat_history[:-1]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Отображение изображений, если они есть
            if message["role"] == "assistant" and "images" in message and message["images"]:
                with st.expander("Релевантные изображения"):
                    cols = st.columns(min(3, len(message["images"])))
                    for i, img in enumerate(message["images"][:3]):
                        with cols[i % 3]:
                            file_path = img.get("file_path", "")
                            if os.path.exists(file_path):
                                try:
                                    # Отображение изображения с адаптивной шириной
                                    st.image(file_path, caption=img.get("title", ""), use_container_width=True)
                                    # Кнопка для скачивания оригинала
                                    with open(file_path, "rb") as file:
                                        mime_type = get_mime_type(file_path)
                                        st.download_button(
                                            label="Скачать изображение",
                                            data=file,
                                            file_name=img.get("file_name", "image"),
                                            mime=mime_type
                                        )
                                except Exception as e:
                                    st.error(f"Не удалось отобразить изображение: {str(e)}")
                            else:
                                st.warning(f"Изображение не найдено: {file_path}")

    # Если есть последнее сообщение, отображаем его отдельно
    if st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]
        # Поскольку последнее сообщение уже отображено через потоковую обработку, нам не нужно его снова отображать

    # Поле ввода для пользователя
    user_input = st.chat_input("Введите ваш вопрос здесь...")
    if user_input:
        handle_user_input(user_input)


def change_model(model_name):
    # Логгирование изменения модели
    import logging
    logger = logging.getLogger("streamlit_app")
    logger.info(f"Смена модели на: {model_name}")

    # Обновление состояния
    st.session_state.selected_model = model_name
    st.success(f"Выбрана модель: {model_name.capitalize()}")


def main():
    # Получаем логгер
    import logging
    logger = logging.getLogger("streamlit_app")

    logger.info("Запуск приложения")

    # Проверяем, аутентифицирован ли пользователь
    if not st.session_state.authenticated:
        login_user()
        return

    # Если пользователь аутентифицирован, показываем основной интерфейс
    st.sidebar.title("1С RAG-ассистент")

    # Добавляем выбор модели LLM в боковую панель
    st.sidebar.subheader("Выбор LLM модели")
    model_options = {
        "anthropic": "Anthropic Claude",
        "deepseek": "Deepseek Chat"
    }

    selected_model = st.sidebar.radio(
        "Выберите модель:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0 if st.session_state.selected_model == "anthropic" else 1
    )

    # Если выбрана новая модель, обновляем состояние
    if selected_model != st.session_state.selected_model:
        logger.info(f"Изменение модели с {st.session_state.selected_model} на {selected_model}")
        change_model(selected_model)

    # Основная страница чата
    chat_page()

    st.sidebar.divider()

    # Добавляем кнопку для сброса истории чата
    if st.sidebar.button("Очистить историю чата"):
        logger.info("Очистка истории чата")
        st.session_state.chat_history = []
        st.rerun()

    # Панель администратора (видна только администраторам)
    if st.session_state.current_user["role"] == "admin":
        st.sidebar.divider()
        st.sidebar.subheader("Панель администратора")

        with st.sidebar.expander("Список пользователей"):
            for username, user_data in USERS.items():
                st.markdown(f"**{user_data['full_name']}** ({username})")
                st.markdown(f"Роль: {user_data['role'].capitalize()}")
                st.divider()

    st.sidebar.divider()
    st.sidebar.caption("© 2025 1С RAG-ассистент")


if __name__ == "__main__":
    main()