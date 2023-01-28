import io
import streamlit as st
from PIL import Image, ImageDraw
import torch

def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите изображение для детекции')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def edit_foto(table_class, image_pil):
    image = image_pil.convert("RGBA")
    image2 = Image.new('RGBA', (image.width, image.height))
    tempDraw = ImageDraw.Draw(image2)
    for row in range(len(table_class)):
        tempDraw.rectangle([(table_class.iloc[row, 0], table_class.iloc[row, 1]),
                            (table_class.iloc[row, 2], table_class.iloc[row, 3])],
                           fill=(0, 0, 255, 150),
                           outline="red")
    image.paste(image2, (0, 0), image2)
    return image

@st.cache
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    return model

def main():
    page_list = ["О проекте", "Детекция"]
    st.sidebar.markdown("# Меню")
    page = st.sidebar.selectbox("Выберите страницу:", page_list)

    if page == page_list[0]:
        st.markdown('## Итоговый проект по теме: "Detection"')
        st.image('logo_DLS.jpg', width=300)
        st.markdown("*Deep Learning (семестр 1, осень 2022): базовый поток*")
        st.markdown('#### Автор работы: [Евгений Сомов](https://t.me/evgeny_somov)')
        st.markdown('В данной работе встроена предобученная модель для детекции'
                    ' [YOLO-v5 от Ultralytics](https://github.com/ultralytics/yolov5).  \n'
                    'Модель настроена для определения людей на фотографиях. Веб-приложение создано '
                    'на базе библиотеки [Streamlit](https://streamlit.io). На следующей странице вы сможете '
                    'протестировать модель детекции.')
    elif page == page_list[1]:
        st.title('Детекция людей на изображениях')
        original_imag_pil = load_image()
        result = st.button('Найти людей на изображении')

        if result and original_imag_pil is not None:
            model = load_model()
            st.text("Поиск людей на изображении")
            result_predict = model(original_imag_pil)
            table_class = result_predict.pandas().xyxy[0]
            table_class = table_class[table_class["class"] == 0]
            image = edit_foto(table_class, original_imag_pil)
            st.image(image)

if __name__ == "__main__":
    main()
