import sys 
sys.path.insert(1, "utils")
import gradio as gr
import pandas as pd
import ui_utils 
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import tempfile
import appriori
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="teal",
).set(
    body_background_fill='*neutral_50',
    body_text_color_subdued='*secondary_950',
    body_text_weight='300',
    background_fill_secondary='*neutral_100',
    background_fill_secondary_dark='*neutral_700',
    link_text_color='*primary_300',
    checkbox_background_color='*primary_700',
    checkbox_background_color_dark='*primary_400',
    checkbox_background_color_focus='*primary_50',
    checkbox_background_color_focus_dark='*primary_200',
    button_border_width='*checkbox_label_border_width',
    button_shadow='*button_shadow_hover',
    button_shadow_active='*button_shadow_hover',
    button_large_radius='*radius_xxl'
)
options = ["Dataset 1", "Dataset 2", "Dataset 3"]
blocks = gr.Blocks(theme=theme,
                   css="""
        .header {
            text-align: center;
            padding: 0px;
            background-color: #f2f2f2;
        }
        .header img {
            max-width: 100%;
            max-height: 100%;
            margin-bottom: 0px;
        }
    """, title="InsightData")

with blocks as demo:

    gr.HTML("""<div class='header'><img src='http://localhost:8000/header.png' alt='Header Image'>     
            </div>""")    
    drop = gr.Dropdown(choices=options, label="Select an option", value="Dataset 1")
    inputs = [
        drop ,
        gr.Radio(["before preprocessing", "after preprocessing"], label="Select an option", value="before preprocessing")
    ]
    description = gr.HTML("""<div style="background-color: #135461; color: white; padding: 15px; border-radius: 10px; text-align: justify;">
                <h2 style="text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; font-weight: bold; color: #cbfbf3;">Comprehensive Soil Fertility Analysis</h2>
                <p>Dataset 1 represents a crucial resource for evaluating soil fertility through static data encompassing soil properties. Each entry in the dataset provides key information on soil characteristics, including nutrient levels such as nitrogen (N), phosphorus (P), potassium (K), soil pH, electrical conductivity (EC), organic carbon (OC), sulfur (S), zinc (Zn), iron (Fe), copper (Cu), manganese (Mn), boron (B), organic matter (OM), and soil fertility. This dataset offers a detailed view of the chemical, physical, and biological properties of the soil, enabling researchers and agronomy experts to analyze soil composition and assess its capacity to support crop growth. This static dataset provides a solid foundation for in-depth analyses aimed at improving agricultural practices, maximizing yields, and promoting sustainable land use.</p>
                </div>
                """)
    with gr.Tab("Show Dataframe"):
        show_df = gr.Button("Show Dataframe")
        show_df.click(fn=ui_utils.opendatafile, inputs=inputs, outputs=gr.DataFrame())
    with gr.Tab("Show Dataframe Informations"):
        with gr.Row():
            with gr.Column():
                show_general_info = gr.Button("General Informations")
            with gr.Column():
                show_col_info = gr.Button("Show columns informations")
            with gr.Column():
                show_cent_trends = gr.Button("Show central trends")
        output_df = gr.DataFrame()
        show_general_info.click(fn=ui_utils.print_info, inputs=inputs, outputs=output_df)
        show_col_info.click(fn=ui_utils.describe_column, inputs=inputs, outputs=output_df)
        show_cent_trends.click(fn=ui_utils.show_cent_trends, inputs=inputs, outputs=output_df)
        show_cent_trends
    with gr.Tab("Visualizations"):
        with gr.Row(visible= True) as vis1:
            with gr.Row():
                display1 = gr.Dropdown(
                    choices=["Boxplots", "Histograms", "Scatterplots"],
                    label="Graphe to visualize",
                )
                with gr.Column(visible=False) as attribut_choice1:
                    choice_attribut = gr.Dropdown(choices=ui_utils.get_attribut_data1(),label="Attribut",value="N")
                    show_box_plot1 = gr.Button("Show Box Plot")
                with gr.Column(visible=False) as hist_choice1:
                    choice_hist_attribute = gr.Dropdown(choices=ui_utils.get_attribut_data1(),label="Attribut",value="N")
                    show_hist1 = gr.Button("Show Histogram")
                with gr.Column(visible=False) as scatter_choice:
                    choice_scatter_attribute_1 = gr.Dropdown(choices=ui_utils.get_attribut_data1(),label="Attribut",value="N")
                    choice_scatter_attribute_2 = gr.Dropdown(choices=ui_utils.get_attribut_data1(),label="Attribut",value="N")
                    show_scatter = gr.Button("Show Scatter Plot")
        with gr.Row(visible=False) as vis2:
            with gr.Row():
                display = gr.Dropdown(
                        choices=[
                            "Boxplots",
                            "Histograms",
                            "Distribution of the total number of confirmed cases and positive tests by ZIPCODE",
                            "Evolution of COVID-19 tests, positive tests, and confirmed cases for a given area",
                            "Distribution of positive COVID-19 cases by area and by year",
                            "Relation between the population and the number of tests conducted",
                            "Top 5 most affected areas by COVID-19",
                            "The ratio between confirmed cases, tests conducted, and positive tests for a given zone during a given period",
                        ],
                        label="Graph to visualize",
                    )
                with gr.Column(visible=False) as areas_choice:
                    choice_area_1 = gr.Dropdown(
                            choices=[
                                "95129", "95128", "95127", "95035","94087", "94086", "94085",
                            ],
                            label="Area",value="95129"
                        )
                    choice_period_1 = gr.Dropdown(
                            choices=[
                                "WEEKLY", "MONTHLY", "YEARLY",
                            ],
                            label="Period",value="WEEKLY"
                    )
                    show_graph = gr.Button("Show Graph")
                with gr.Column(visible=False) as period_choice:
                    choice_period_2 = gr.Dropdown(choices=ui_utils.get_periode(),label="Period",value=21)
                    show_graph2 = gr.Button("Show Graph")
                with gr.Column(visible=False) as attribut_choice:
                    choice_area_2 = gr.Dropdown(
                            choices=[
                                "95129", "95128", "95127", "95035","94087", "94086", "94085",
                            ],
                            label="Area",value="95129"
                        )
                    choice_attribut_1 = gr.Dropdown(choices=ui_utils.get_attribut_data2(),label="Attribut",value="case count")
                    show_box_plot = gr.Button("Show Box Plot")
                with gr.Column(visible=False) as hist_choice:
                    choice_attribut_2 = gr.Dropdown(choices=ui_utils.get_attribut_data2(),label="Attribut",value="case count")
                    show_hist = gr.Button("Show Histogram")
        with gr.Row(visible=False) as vis3:
            attribut_choice3 = gr.Dropdown(choices=ui_utils.get_attribut_data3(),label="Attribut",value="Temperature")
            show_hist3 = gr.Button("Show Histogram")

        with gr.Row(visible=False) as image_field:
            graph2 = gr.Image()
    with gr.Row(visible=False) as appriori:
        with gr.Column():
            with gr.Row():
                min_conf = gr.Number(maximum=1,minimum=0,label="Min_Conf",value=0.5)
                min_sup = gr.Number(minimum=0,precision=0,label="Min_Sup",value=3)
                min_cor = gr.Number(maximum=1,minimum=0,label="Min_Corelation",value=0.3)
                mesure = gr.Dropdown(choices=["confidence","lift","cosine"],label="Used mesure",value="confidence")
            with gr.Row():
                apply = gr.Button("Apply Appriori Algorithm")
            with gr.Row(visible=False) as df_appriori:
                df3 = gr.DataFrame()
            with gr.Row(visible=False) as recommandation:
                with gr.Column():
                    with gr.Row():
                        gr.HTML("""<div style="background-color: #135461; color: white; padding: 15px; border-radius: 10px; text-align: justify;">
                        <h2 style="text-align: center; font-family: 'Arial', sans-serif; font-size: 24px; font-weight: bold; color: #cbfbf3;"> Would you like a recommendation ? </h2>
                        <p>Let us assist you in making informed decisions. Provide us with your information, and we will take care of delivering the best recommendations for you.</p>
                        </div>
                        """)
                    with gr.Row():  
                        with gr.Column():  
                            with gr.Row():
                                soil = gr.Dropdown(choices=['Clayey', 'laterite', 'silty clay', 'sandy', 'coastal', 'clay loam' ,'alluvial'],label="Soil Type",value="Clayey") 
                                crop = gr.Dropdown(choices=['rice', 'Coconut'],label="Crop Type",value="rice")
                                rain = gr.Dropdown(choices=['Very Low', 'Low', 'Moderate', 'High'],label="Rainfall",value='Very Low')
                                show_rec = gr.Button("Show Recommendation")
                            with gr.Row():
                                rec = gr.Textbox(interactive=False,label="Recommendation")


   
   
   
   
   
   
   
   
   
   
   
   
   
   
    def set_visible(data,pre):
        if data == "Dataset 2":
            return {vis2 : gr.Row(visible = True),appriori : gr.Row(visible=False),vis3: gr.Row(visible=False) ,vis1 : gr.Row(visible=False),description : ui_utils.dataset_description(data,pre),image_field : gr.Row(visible = False)}
        if data == "Dataset 1":
            return{vis2 : gr.Row(visible = False),appriori : gr.Row(visible=False),vis3: gr.Row(visible=False),vis1 : gr.Row(visible=True),description : ui_utils.dataset_description(data,pre),image_field : gr.Row(visible = False)}
        if  data == "Dataset 3":
            return{vis2 : gr.Row(visible = False),appriori : gr.Row(visible=True),vis3: gr.Row(visible=True),vis1 : gr.Row(visible=False),description : ui_utils.dataset_description(data,pre),image_field : gr.Row(visible = True)}
    def graphe2_selction(choice,data,pre):
        if choice == "Boxplots":
            return{attribut_choice : gr.Column(visible = True),hist_choice : gr.Column(visible=False), areas_choice : gr.Column(visible = False),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = True)}
        if choice == "Histograms":
            return{hist_choice : gr.Column(visible = True),attribut_choice : gr.Column(visible=False) ,areas_choice : gr.Column(visible = False),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = True)}
        if choice == "Distribution of the total number of confirmed cases and positive tests by ZIPCODE":
            if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to see the graph")
            else:
                return {graph2 : ui_utils.graphe_2_1(data,pre),hist_choice: gr.Column(visible=False),attribut_choice : gr.Column(visible=False) ,areas_choice : gr.Column(visible = False),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = True)}
        elif choice == "Evolution of COVID-19 tests, positive tests, and confirmed cases for a given area":
            if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to see the graph")
            else:
                return {attribut_choice : gr.Column(visible=False) ,hist_choice: gr.Column(visible=False), areas_choice : gr.Column(visible = True),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = False)}
        elif choice == "Distribution of positive COVID-19 cases by area and by year":
            if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to see the graph")
            else:
                return {graph2: ui_utils.graphe_2_3(data,pre),attribut_choice : gr.Column(visible=False) ,hist_choice: gr.Column(visible=False),areas_choice : gr.Column(visible = False),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = True)}
        elif choice == "Relation between the population and the number of tests conducted":
            if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to see the graph")
            else:    
                return {graph2: ui_utils.graphe_2_4(data,pre),attribut_choice : gr.Column(visible=False) ,hist_choice: gr.Column(visible=False),areas_choice : gr.Column(visible = False),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = True)}
        elif choice == "Top 5 most affected areas by COVID-19":
            if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to see the graph")
            else:
                return {graph2: ui_utils.graphe_2_5(data,pre),attribut_choice : gr.Column(visible=False) ,hist_choice: gr.Column(visible=False),areas_choice : gr.Column(visible = False),period_choice : gr.Column(visible = False),image_field : gr.Row(visible = True)}
        elif choice == "The ratio between confirmed cases, tests conducted, and positive tests for a given zone during a given period":
            if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to see the graph")
            else:
                return {period_choice : gr.Column(visible = True),attribut_choice : gr.Column(visible=False) ,hist_choice: gr.Column(visible=False),areas_choice : gr.Column(visible = False),image_field : gr.Row(visible = False)}
    def show_graph2_2 (data,pre,area,period):
        return{graph2: ui_utils.graphe_2_2(data,pre,area,period), image_field : gr.Row(visible = True)}
    def show_graph2_6 (data,pre,period):
        return{graph2: ui_utils.graphe_2_6(data,pre,period), image_field : gr.Row(visible = True)}
    def graph1_selection(choice):
        if choice == "Boxplots":
            return{attribut_choice1 : gr.Column(visible = True),scatter_choice : gr.Column(visible=False),hist_choice1 :gr.Column(visible=False) ,image_field : gr.Row(visible = True)}
        elif choice == "Histograms":
            return{attribut_choice1 : gr.Column(visible = False),scatter_choice : gr.Column(visible=False),hist_choice1 : gr.Column(visible=True),image_field : gr.Row(visible = True)}
        elif choice == "Scatterplots":
            return{attribut_choice1 : gr.Column(visible = False),scatter_choice : gr.Column(visible=True),hist_choice1 : gr.Column(visible=False),image_field : gr.Row(visible = True)}
    def check_sctter(data,pre,attribut_1,attribut_2):
        if attribut_1 == attribut_2:
            return gr.Info("Please select two different attributes")
        else:
            return {graph2: ui_utils.scatter_plot(data,pre,attribut_1,attribut_2)}
    def apply_appriori(data,pre, min_conf,min_sup,min_cor,mesure):
        if pre == "before preprocessing":
                return gr.Info("The dataset is not preprocessed yet, please select the option 'after preprocessing' to apply the appriori algorithme")
        else:
            return {df_appriori : gr.Row(visible = True),recommandation : gr.Row(visible = True), df3 : ui_utils.show_assoc_rules(min_conf,min_sup,min_cor,mesure)}
    inputs[0].select(fn=set_visible, inputs = inputs,outputs=[description,vis2,vis1,image_field,vis3,appriori])
    display.select(fn=graphe2_selction,inputs=[display]+inputs,outputs=[graph2,areas_choice,period_choice,image_field,attribut_choice,hist_choice])
    show_graph.click(fn=show_graph2_2,inputs=inputs+[choice_area_1,choice_period_1],outputs=[graph2,image_field])
    show_graph2.click(fn=show_graph2_6,inputs=inputs+[choice_period_2],outputs=[graph2,image_field])
    show_box_plot.click(fn=ui_utils.box_plot,inputs=inputs+[choice_area_2,choice_attribut_1],outputs=[graph2])
    show_hist.click(fn=ui_utils.histogram,inputs=inputs+[choice_attribut_2],outputs=[graph2])
    display1.select(fn=graph1_selection,inputs=display1,outputs=[graph2,attribut_choice1,image_field,hist_choice1,scatter_choice])
    show_hist1.click(fn=ui_utils.histogram,inputs=inputs+[choice_hist_attribute],outputs=[graph2])
    show_box_plot1.click(fn=ui_utils.box_plot1,inputs=inputs+[choice_attribut],outputs=[graph2])
    show_scatter.click(fn=check_sctter,inputs=inputs+[choice_scatter_attribute_1,choice_scatter_attribute_2],outputs=[graph2])
    show_hist3.click(fn=ui_utils.histogram,inputs=inputs+[attribut_choice3],outputs=[graph2])
    apply.click(fn=apply_appriori,inputs=inputs+[min_conf,min_sup,min_cor,mesure],outputs=[df_appriori,recommandation,df3])
    show_rec.click(fn=ui_utils.recommendation,inputs=[soil,crop,rain,min_conf,min_sup,min_cor,mesure],outputs=[rec])
demo.launch()
# favicon_path = "http://localhost:8000/icone.ico"