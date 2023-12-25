import customtkinter
import torch
import numpy as np
import joblib

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 516),
            torch.nn.ReLU(),
            torch.nn.Linear(516, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)

def validate_int(input_str):
    try:
        int(input_str)
        return True
    except ValueError:
        return False

def validate(input_str):
    if len(fst.get())!=0:
        try:
            int(input_str)
            return True
        except ValueError:
            return False
    else:
        return False
    

def create_label_entry_pair(root, row, text, validate_int_var=None):
    label = customtkinter.CTkLabel(root, text=text, font=('calibre', 12, 'bold'))
    label.grid(row=row, column=0, padx=(10, 10), pady=(10, 5), sticky='w')

    entry = customtkinter.CTkEntry(root, font=('calibre', 12, 'normal'))
    if validate_int_var:
        entry.configure(validate='key', validatecommand=(root.register(validate_int_var), '%P'))
    entry.grid(row=row, column=1, padx=(10, 10), pady=(10, 5), sticky='e')

    return entry

def predict_price():
    total_sqf = int(snd.get()) + int(fst.get()) + int(b.get())
    living = int(snd.get()) + int(fst.get())
    samples = [round(slider.get()), b.get(), fst.get(), living, cap.get(), ar.get(), total_sqf, bathrooms.get()]
    samples = torch.tensor(scaler.transform(np.double(samples).reshape(1, -1)).reshape(-1, 1))

    ml = NeuralNetwork()
    device = 'cpu'
    ml.load_state_dict(torch.load('model.pt', map_location=device))
    ml.eval()

    samples = samples.squeeze()
    pred = ml(samples.float())
    pred = scaler_y.inverse_transform(pred.detach().numpy().reshape(1, -1))
    prediction_label.configure(text=f"Predicted Price: ${pred[0][0]:,.2f}", font=('calibre', 14, 'bold'), text_color='green')

def update_slider_value(value):
    slider_value_label.configure(text=f"{round(value)}")

# Load scaler and model
scaler = joblib.load('scaler.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# GUI Setup
root = customtkinter.CTk()
root.geometry("500x600")
root.title("House Price Prediction")

customtkinter.set_appearance_mode("dark")

# Housing Zone Dropdown
zone_label = customtkinter.CTkLabel(root, text='Housing Zone', font=('calibre', 14, 'bold'))
zone_label.grid(row=0, column=0, padx=(10, 10), pady=(10, 5), sticky='w')
zone_combobox = customtkinter.CTkComboBox(master=root,
                                           values=["Agriculture", "Commercial", "Floating Village Residential",
                                                   "Industrial", "Residential High Density", "Residential Low Density",
                                                   "Residential Low Density Park", "Residential Medium Density"])
zone_combobox.grid(row=0, column=1, padx=(10, 10), pady=(10, 5), sticky='e')

# House Quality Slider
slider_label = customtkinter.CTkLabel(root, text='House Quality', font=('calibre', 14, 'bold'))
slider_label.grid(row=1, column=0, padx=(10, 10), pady=(10, 5), sticky='w')
slider = customtkinter.CTkSlider(master=root, from_=1, to=10, command=lambda value: update_slider_value(value), number_of_steps=9)
slider.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)
slider.grid(row=1, column=1, padx=(10, 10), pady=(10, 5), sticky='e')
slider_value_label = customtkinter.CTkLabel(root, text="", font=('calibre', 12, 'italic'))
slider_value_label.grid(row=1, column=2, columnspan=2, pady=(5, 5), sticky='w')

# Entry Fields
cap = create_label_entry_pair(root, 2, 'Garage Capacity', validate_int)
ar = create_label_entry_pair(root, 3, 'Garage Size (sqft)', validate_int)
b = create_label_entry_pair(root, 4, 'Basement Size (sqft)', validate_int)
fst = create_label_entry_pair(root, 5, 'First Floor Size (sqft)', validate_int)
snd = create_label_entry_pair(root, 6, 'Second Floor Size (sqft)', validate)
bathrooms = create_label_entry_pair(root, 7, 'Total Bathrooms', validate_int)

# Prediction Result Label
prediction_label = customtkinter.CTkLabel(master=root, text="", font=('calibre', 16, 'bold'),text_color='green')
prediction_label.grid(row=10, column=0, columnspan=2, pady=(10, 10), sticky='w')

#  Predict Button
predict_button = customtkinter.CTkButton(master=root, text="Predict House Price", command=predict_price, font=('calibre', 14, 'bold'))
predict_button.grid(row=8, column=1, columnspan=2, pady=(10, 10), padx=(10, 10), sticky='e')

root.mainloop()