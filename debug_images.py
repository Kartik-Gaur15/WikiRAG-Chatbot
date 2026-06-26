import wikipedia
wikipedia.set_user_agent("WikiRAG-Chatbot/1.0 (test)")

page = wikipedia.page("IPhone 15", auto_suggest=False, redirect=True)
images = page.images
print(f"Total images found: {len(images)}")
for url in images[:15]:
    print(" -", url)
