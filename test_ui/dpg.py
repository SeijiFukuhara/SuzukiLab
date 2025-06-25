import dearpygui.dearpygui as dpg

dpg.create_context()
with dpg.window(label="Hello"):
    dpg.add_text("Dear PyGui is working!")
dpg.create_viewport(title='DPG Test', width=400, height=200)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
