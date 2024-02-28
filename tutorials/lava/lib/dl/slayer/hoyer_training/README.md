
| Directory    | File           | Description / Modifications                                                                                                     |
|--------------|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| `block`      | `__init__.py`  | Add `'cuba_hoyer'`.                                                                                                              |
|              | `cuba_hoyer.py`| Use `cuba.HoyerNeuron` instead of `cuba.Neuron` in the `AbastractCubaHoyer` class; inherit this for other layers.                |
|              | `base.py`      | Add `self.synapse.pre_hook_fx = self.neuron.quantize_8bit if self.synapse.pre_hook_fx is None else self.synapse.pre_hook_fx` in `export_hdf5` for `Dense` and `Conv`. |
| `neuron`     | `__init__.py`  | Add `'cuda_hoyer'`.                                                                                                              |
|              | `cuba_hoyer.py`| Define the new class `HoyerNeuron`, and add the calculation for Hoyer Ext in the forward part.                                   |
| `spike`      | `spike.py`     | Add the new class `HoyerSpike`, which has the same forward part but a different forward part.                                    |
| **Other Questions** |      |                                                                                                                                  |
|              |                | The forward part for maxpooling layer.                                                                                           |
|              |                | The last dense layer without spiking.                                                                                            |
|              |                | The delay.                                                                                            |